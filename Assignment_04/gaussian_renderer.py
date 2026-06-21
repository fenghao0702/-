import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
        
        # 2. Get depths before projection for proper sorting and clipping
        depths = cam_points[:, 2].clamp(min=1.)  # (N, )
        
        # 3. Project to screen space using camera intrinsics
        screen_points = cam_points @ K.T  # (N, 3)
        means2D = screen_points[..., :2] / screen_points[..., 2:3] # (N, 2)
        
        # Compute Jacobian of perspective projection
        # Projection: u = fx*x/z + cx,  v = fy*y/z + cy
        # ∂u/∂(x,y,z) = [fx/z, 0, -fx*x/z²]
        # ∂v/∂(x,y,z) = [0, fy/z, -fy*y/z²]
        J_proj = torch.zeros((N, 2, 3), device=means3D.device)
        fx = K[0, 0]
        fy = K[1, 1]
        cam_x = cam_points[:, 0]
        cam_y = cam_points[:, 1]
        cam_z = cam_points[:, 2].clamp(min=1.0)  # match depth clamp to avoid extreme Jacobian
        cam_z_sq = cam_z * cam_z

        J_proj[:, 0, 0] = fx / cam_z
        J_proj[:, 0, 2] = -fx * cam_x / cam_z_sq
        J_proj[:, 1, 1] = fy / cam_z
        J_proj[:, 1, 2] = -fy * cam_y / cam_z_sq

        # Transform covariance to camera space: Σ_cam = R Σ Rᵀ
        R_exp = R.unsqueeze(0)  # (1,3,3) broadcasts to (N,3,3)
        covs_cam = R_exp @ covs3d @ R_exp.transpose(1, 2)  # (N, 3, 3)
        
        # Project to 2D
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))  # (N, 2, 2)
        
        return means2D, covs2D, depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)
        
        # Compute Gaussian values: f(x;μ,Σ) = 1/(2π√|Σ|) * exp(-½(x-μ)ᵀ Σ⁻¹ (x-μ))
        # Explicit 2×2 inverse for efficiency: Σ = [[a,b],[b,d]], Σ⁻¹ = 1/det * [[d,-b],[-b,a]]
        a = covs2D[:, 0, 0]  # (N,)
        b = covs2D[:, 0, 1]  # (N,) — symmetric, so cov[:,0,1] == cov[:,1,0]
        d = covs2D[:, 1, 1]  # (N,)
        det = a * d - b * b  # (N,)
        inv_det = 1.0 / det.clamp(min=1e-9)

        dx_x = dx[..., 0]  # (N, H, W)
        dx_y = dx[..., 1]  # (N, H, W)

        # Quadratic form: (x-μ)ᵀ Σ⁻¹ (x-μ) = (d·dx² − 2b·dx·dy + a·dy²) / det
        quad = inv_det.view(N, 1, 1) * (
            d.view(N, 1, 1) * dx_x * dx_x
            - 2.0 * b.view(N, 1, 1) * dx_x * dx_y
            + a.view(N, 1, 1) * dx_y * dx_y
        )

        power = -0.5 * quad
        power = torch.clamp(power, max=0.0)  # numerical safety
        norm = 1.0 / (2.0 * torch.pi * torch.sqrt(det.clamp(min=1e-9)) + 1e-9)
        gaussian = norm.view(N, 1, 1) * torch.exp(power)  # (N, H, W)

        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]
        
        # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 2. Depth mask
        valid_mask = (depths > 1.) & (depths < 50.0)  # (N,)
        
        # 3. Sort by depth
        indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[ indices]       # (N, 3)
        opacities = opacities[indices] # (N, 1)
        valid_mask = valid_mask[indices] # (N,)
        
        # 4. Compute gaussian values
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # 5. Apply valid mask
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)
        
        # 6. Alpha composition setup
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # Alpha-blending weights: w_i = α_i * Π_{j<i} (1 − α_j)
        T = torch.cumprod(1.0 - alphas + 1e-10, dim=0)  # cumulative transmission
        T = torch.cat([torch.ones(1, self.H, self.W, device=alphas.device), T[:-1]], dim=0)
        weights = alphas * T  # (N, H, W)
        
        # 8. Final rendering
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        return rendered
