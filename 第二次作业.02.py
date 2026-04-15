import torch.nn as nn
import torch

class FullyConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: 逐渐减少尺寸，增加通道
        # 输入: (3, 256, 256)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 128, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 64, 64)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (256, 32, 32)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # -> (512, 16, 16)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder: 逐渐增加尺寸，减少通道
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # -> (256, 32, 32)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 64, 64)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 128, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # -> (3, 256, 256)
            nn.Tanh()  # 将输出限制在 [-1, 1]
        )

    def forward(self, x):
        # Encoder forward
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # Decoder forward
        d1 = self.deconv1(x4)
        d2 = self.deconv2(d1)
        d3 = self.deconv3(d2)
        output = self.deconv4(d3)

        return output