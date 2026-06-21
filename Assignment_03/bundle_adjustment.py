"""
Bundle Adjustment with PyTorch — Task 1
=======================================
Recover 3D point coordinates, camera extrinsics (R, T), and shared focal length f
from 2D observations across 50 views using gradient-based optimization.

Reference: README.md — Coordinate System & Initialization, Projection Formula
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch3d.transforms import euler_angles_to_matrix
import matplotlib.pyplot as plt
import os
import time

# ==============================================================================
# Section 1: Constants
# ==============================================================================

IMG_W, IMG_H = 1024, 1024
CX, CY = 512.0, 512.0
N_VIEWS = 50
N_POINTS = 20000
D_INIT = 2.5           # initial camera distance (object → camera)
FOV_INIT_DEG = 60.0     # initial field-of-view estimate for f initialization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


# ==============================================================================
# Section 2: Data Loading
# ==============================================================================

def load_data(data_dir="data"):
    """
    Load 2D observations and 3D point colors.

    Returns:
        obs_dict:   dict[str, np.ndarray] — 50 keys "view_000".."view_049",
                    each (20000, 3) with columns [x, y, visibility]
        colors:     np.ndarray (20000, 3) — RGB colors in [0, 1]
    """
    obs_path = os.path.join(data_dir, "points2d.npz")
    colors_path = os.path.join(data_dir, "points3d_colors.npy")

    obs_dict = dict(np.load(obs_path))
    colors = np.load(colors_path)

    # sanity checks
    assert len(obs_dict) == N_VIEWS, f"Expected {N_VIEWS} views, got {len(obs_dict)}"
    sample_shape = list(obs_dict.values())[0].shape
    assert sample_shape == (N_POINTS, 3), f"Expected ({N_POINTS}, 3), got {sample_shape}"
    assert colors.shape == (N_POINTS, 3), f"Expected ({N_POINTS}, 3), got {colors.shape}"

    return obs_dict, colors


def precompute_visibility(obs_dict, device=DEVICE):
    """
    For each view, extract indices and (x, y) of visible points.
    Returns PADDED tensors for batched GPU processing — single kernel launch per iter.

    Returns:
        all_point_idx: (50, MAX_VIS) LongTensor  — padded, invalid entries set to 0
        all_obs_xy:    (50, MAX_VIS, 2) FloatTensor — padded, invalid entries set to 0
        valid_mask:    (50, MAX_VIS) BoolTensor — True where observation is real
        vis_counts:    (50,) LongTensor — number of visible points per view
    """
    vis_counts = []
    point_idx_list = []
    obs_xy_list = []

    for i in range(N_VIEWS):
        key = f"view_{i:03d}"
        obs = obs_dict[key]                     # (20000, 3): [x, y, visibility]
        mask = obs[:, 2] == 1.0
        indices = np.where(mask)[0]
        vis_counts.append(len(indices))
        point_idx_list.append(torch.from_numpy(indices).long())
        obs_xy_list.append(torch.from_numpy(obs[indices, :2]).float())

    # Pad to max visible count for batched processing
    max_vis = max(vis_counts)
    N = len(vis_counts)

    all_point_idx = torch.zeros(N, max_vis, dtype=torch.long, device=device)
    all_obs_xy = torch.zeros(N, max_vis, 2, device=device)
    valid_mask = torch.zeros(N, max_vis, dtype=torch.bool, device=device)

    for i in range(N):
        n = vis_counts[i]
        all_point_idx[i, :n] = point_idx_list[i].to(device)
        all_obs_xy[i, :n] = obs_xy_list[i].to(device)
        valid_mask[i, :n] = True

    vis_counts = torch.tensor(vis_counts, dtype=torch.long, device=device)
    total_vis = vis_counts.sum().item()

    # print statistics
    print(f"Total visible observations: {total_vis} / {N_VIEWS * N_POINTS} "
          f"({100 * total_vis / (N_VIEWS * N_POINTS):.1f}%)")
    print(f"Max visible per view: {max_vis}  (padding overhead: "
          f"{100 * (N * max_vis - total_vis) / total_vis:.1f}%)")
    for i in [0, 12, 25, 37, 49]:
        print(f"  {f'view_{i:03d}'}: {vis_counts[i].item()} visible points")

    return all_point_idx, all_obs_xy, valid_mask, vis_counts


# ==============================================================================
# Section 3: Euler Angles → Rotation Matrix (pytorch3d)
# ==============================================================================
# Uses pytorch3d.transforms.euler_angles_to_matrix (imported at top).
# If this fails with ImportError, run: pip install pytorch3d


# ==============================================================================
# Section 4: Projection Function
# ==============================================================================

def project(points_3d, R, T, f, cx=CX, cy=CY):
    """
    Project 3D points to 2D pixel coordinates using the pinhole camera model.

    Camera model (from README):
        [Xc, Yc, Zc]^T = R @ [X, Y, Z]^T + T
        u = -f * Xc / Zc + cx          ← negative sign because Zc < 0
        v =  f * Yc / Zc + cy

    Args:
        points_3d: (N, 3) — 3D point coordinates in world frame
        R:         (3, 3) or (B, 3, 3) — rotation matrix
        T:         (3,)   or (B, 3)    — translation vector
        f:         scalar — focal length (pixels)
        cx, cy:    float — principal point

    Returns:
        u, v: each (N,) or (B, N) — predicted pixel coordinates
    """
    # camera coordinates: Xc = R @ P^T + T
    # points_3d: (N, 3), R: (3, 3) → Xc: (N, 3)
    Xc = points_3d @ R.T + T          # (N, 3)

    Xc_x, Xc_y, Xc_z = Xc[..., 0], Xc[..., 1], Xc[..., 2]

    # prevent division by zero — add small epsilon with sign preservation
    eps = torch.where(Xc_z < 0,
                      torch.tensor(-1e-8, device=Xc_z.device, dtype=Xc_z.dtype),
                      torch.tensor(1e-8, device=Xc_z.device, dtype=Xc_z.dtype))
    Zc_safe = Xc_z + eps

    # focal length must be positive
    f_abs = torch.abs(f)

    u = -f_abs * Xc_x / Zc_safe + cx   # ← negative sign (READMD Eq.)
    v =  f_abs * Xc_y / Zc_safe + cy

    return u, v


# ==============================================================================
# Section 5: Parameter Initialization
# ==============================================================================

def init_parameters(device=DEVICE):
    """
    Initialize all learnable parameters following README recommendations.

    Returns:
        f, euler_angles, T, points_3d — all as nn.Parameter
    """
    torch.manual_seed(SEED)

    # Focal length: estimate from a reasonable FoV
    # f = H / (2 * tan(fov / 2))
    f_init = IMG_H / (2.0 * np.tan(np.radians(FOV_INIT_DEG) / 2.0))
    f = nn.Parameter(torch.tensor([f_init], device=device))
    print(f"Initial f: {f_init:.2f} px  (FoV ~ {FOV_INIT_DEG:.0f} deg)")

    # Euler angles: zeros → identity rotation (all cameras face +Z initially)
    euler_angles = nn.Parameter(torch.zeros(N_VIEWS, 3, device=device))

    # Translations: all cameras on +Z side, distance D_INIT from object
    T_init = torch.zeros(N_VIEWS, 3, device=device)
    T_init[:, 2] = -D_INIT
    T = nn.Parameter(T_init)

    # 3D points: random small values near origin
    points_3d = nn.Parameter(torch.randn(N_POINTS, 3, device=device) * 0.1)

    print(f"Parameters initialized: f=({1},), euler_angles=({N_VIEWS}, 3), "
          f"T=({N_VIEWS}, 3), points_3d=({N_POINTS}, 3)")
    print(f"Device: {device}")

    return f, euler_angles, T, points_3d


# ==============================================================================
# Section 6: Optimization Loop
# ==============================================================================

def run_optimization(f, euler_angles, T, points_3d, all_point_idx, all_obs_xy,
                     valid_mask, vis_counts, max_iter=2000, log_every=100,
                     device=DEVICE):
    """
    Run the Bundle Adjustment optimization — FULLY BATCHED (no Python for-loop).

    Uses padded tensors so all 50 views are projected in a SINGLE GPU kernel launch.

    Args:
        f, euler_angles, T, points_3d: learnable parameters (nn.Parameter)
        all_point_idx: (50, MAX_VIS) LongTensor — padded point indices
        all_obs_xy:    (50, MAX_VIS, 2) FloatTensor — padded observations
        valid_mask:    (50, MAX_VIS) BoolTensor — True = real observation
        vis_counts:    (50,) LongTensor — real count per view

    Returns:
        loss_history: list of (iteration, RMSE_pixels)
    """
    optimizer = Adam([
        {'params': [f],            'lr': 1e-3},
        {'params': [euler_angles], 'lr': 5e-4},
        {'params': [T],            'lr': 5e-3},
        {'params': [points_3d],    'lr': 5e-3},
    ])

    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                   factor=0.5, patience=300,
                                   min_lr=1e-7)

    loss_history = []
    t_start = time.time()

    # Pre-compute indices tensors (never changes)
    total_valid = vis_counts.sum().float()

    for it in range(max_iter):
        optimizer.zero_grad()

        # ---- FULLY BATCHED projection ----
        # all_point_idx: (50, MAX_VIS)
        # points_3d:     (20000, 3)
        # → pts:         (50, MAX_VIS, 3)   — gather all visible 3D points at once
        pts = points_3d[all_point_idx]      # (50, MAX_VIS, 3)

        # R_all: (50, 3, 3)   T: (50, 3)
        R_all = euler_angles_to_matrix(euler_angles, convention="XYZ")  # (50, 3, 3)

        # Xc = pts @ R^T + T    → batched: (50, MAX_VIS, 3)
        Xc = torch.bmm(pts, R_all.transpose(-2, -1)) + T.unsqueeze(1)

        Xc_x, Xc_y, Xc_z = Xc[..., 0], Xc[..., 1], Xc[..., 2]

        # Safe division
        Zc_safe = Xc_z + torch.where(
            Xc_z < 0,
            torch.tensor(-1e-8, device=device, dtype=Xc_z.dtype),
            torch.tensor(1e-8, device=device, dtype=Xc_z.dtype)
        )
        f_abs = torch.abs(f)

        # Projection (batched)
        u_pred = -f_abs * Xc_x / Zc_safe + CX   # (50, MAX_VIS)
        v_pred =  f_abs * Xc_y / Zc_safe + CY   # (50, MAX_VIS)

        # ---- Masked MSE loss ----
        sq_errors = (u_pred - all_obs_xy[..., 0]) ** 2 + \
                    (v_pred - all_obs_xy[..., 1]) ** 2   # (50, MAX_VIS)

        # Per-view mean, then weighted average
        masked_sq = sq_errors * valid_mask.float()
        per_view_sum = masked_sq.sum(dim=1)               # (50,)
        per_view_mean = per_view_sum / vis_counts.float()  # (50,)
        mean_loss = (per_view_sum).sum() / total_valid     # scalar — weight by vis count

        mean_loss.backward()

        optimizer.step()
        scheduler.step(mean_loss.detach())

        # Logging
        if it % log_every == 0 or it == max_iter - 1:
            rmse = torch.sqrt(mean_loss).item()
            loss_history.append((it, rmse))
            elapsed = time.time() - t_start
            lr_current = optimizer.param_groups[0]['lr']
            f_val = f_abs.item()
            print(f"[{it:5d}] RMSE: {rmse:.4f} px  |  f: {f_val:.2f}  |  "
                  f"LR: {lr_current:.2e}  |  time: {elapsed:.0f}s")

        # Early stopping
        if optimizer.param_groups[0]['lr'] < 1e-7:
            print(f"LR below 1e-7 at iteration {it}, stopping early.")
            break

    t_total = time.time() - t_start
    print(f"\nOptimization finished. Total time: {t_total:.0f}s ({t_total/60:.1f} min)")

    return loss_history


# ==============================================================================
# Section 7: Output — Loss Curve & OBJ Export
# ==============================================================================

def plot_loss_curve(loss_history, save_path="loss_curve.png"):
    """Plot RMSE vs iteration (log scale)."""
    iterations = [it for it, _ in loss_history]
    rmses = [rmse for _, rmse in loss_history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, rmses, 'b-', linewidth=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Reprojection Error (RMSE, pixels)')
    ax.set_title('Bundle Adjustment — Convergence')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Annotate final value
    final_rmse = rmses[-1]
    ax.annotate(f'{final_rmse:.2f} px',
                xy=(iterations[-1], final_rmse),
                xytext=(iterations[-1] * 0.85, final_rmse * 1.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to: {save_path}")


def save_colored_obj(filepath, points, colors):
    """
    Export 3D point cloud as OBJ with per-vertex colors.

    OBJ colored vertex format:  v x y z r g b
    where r, g, b are in [0, 1].

    Args:
        filepath: output .obj path
        points:   (N, 3) numpy array — 3D coordinates
        colors:   (N, 3) numpy array — RGB in [0, 1]
    """
    with open(filepath, 'w') as f:
        f.write("# Bundle Adjustment — Reconstructed 3D Point Cloud\n")
        f.write(f"# {len(points)} vertices\n")
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}\n")

    file_size = os.path.getsize(filepath) / 1024
    print(f"OBJ saved to: {filepath} ({len(points)} vertices, {file_size:.1f} KB)")


# ==============================================================================
# Section 8: Sanity Check
# ==============================================================================

def sanity_check():
    """
    Verify the projection formula with a known configuration.
    Point at origin [0,0,0], R=I, T=[0,0,-d] → should project to (cx, cy).
    """
    p = torch.tensor([[0.0, 0.0, 0.0]])
    R = torch.eye(3)
    T = torch.tensor([0.0, 0.0, -D_INIT])
    f = torch.tensor([887.0])

    u, v = project(p, R, T, f)
    print(f"\nSanity check: origin point, R=I, T=[0,0,{-D_INIT}]:")
    print(f"  Predicted (u, v) = ({u.item():.4f}, {v.item():.4f})")
    print(f"  Expected  (u, v) = ({CX}, {CY})")
    assert abs(u.item() - CX) < 1e-4, f"u mismatch: {u.item()} != {CX}"
    assert abs(v.item() - CY) < 1e-4, f"v mismatch: {v.item()} != {CY}"
    print("  [OK] Passed!")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 60)
    print("Bundle Adjustment — Task 1")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading data...")
    obs_dict, colors = load_data("data")

    # 2. Precompute visibility (padded for batched GPU)
    print("\n[2/6] Precomputing visibility masks...")
    all_point_idx, all_obs_xy, valid_mask, vis_counts = precompute_visibility(obs_dict)

    # 3. Initialize parameters
    print("\n[3/6] Initializing parameters...")
    f, euler_angles, T, points_3d = init_parameters()

    # 4. Sanity check
    print("\n[4/6] Running sanity check...")
    sanity_check()

    # 5. Optimize
    if DEVICE.type == "cuda":
        max_iter = 10000
        torch.cuda.synchronize()
    else:
        max_iter = 5000
    print(f"\n[5/6] Starting optimization ({max_iter} iterations on {DEVICE})...")
    loss_history = run_optimization(
        f, euler_angles, T, points_3d,
        all_point_idx, all_obs_xy, valid_mask, vis_counts,
        max_iter=max_iter, log_every=100
    )

    # 6. Output
    print("\n[6/6] Generating outputs...")
    plot_loss_curve(loss_history, "loss_curve.png")

    final_points = points_3d.detach().cpu().numpy()
    save_colored_obj("reconstruction.obj", final_points, colors)

    # Final report
    f_final = torch.abs(f).item()
    fov_final = 2.0 * np.arctan(IMG_H / (2.0 * f_final))
    print(f"\n{'=' * 60}")
    print(f"Final Results:")
    print(f"  Focal length f: {f_final:.2f} px  (FoV ~ {np.degrees(fov_final):.1f} deg)")
    print(f"  Final RMSE:     {loss_history[-1][1]:.4f} px")
    print(f"  Output files:   loss_curve.png, reconstruction.obj")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
