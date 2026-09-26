import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskCNN(nn.Module):
    def __init__(self, num_crops=3, num_health=2):
        super().__init__()
        # ----- 합성곱 층 -----
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 3  -> 16 채널
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16 -> 32 채널
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 -> 64 채널

        # ----- 두 개의 출력 머리 -----
        self.crop_head = nn.Linear(64, num_crops)     # 작물 종류
        self.health_head = nn.Linear(64, num_health)  # 건강 상태

    def forward(self, x):
        # x: (B, 3, 128, 128)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # (B, 16, 64, 64)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (B, 32, 32, 32)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # (B, 64, 16, 16)

        x = F.adaptive_avg_pool2d(x, 1)  # (B, 64, 1, 1)  각 채널의 평균
        feat = torch.flatten(x, 1)       # (B, 64)

        crop_out = self.crop_head(feat)      # (B, 3)
        health_out = self.health_head(feat)  # (B, 2)
        return crop_out, health_out
