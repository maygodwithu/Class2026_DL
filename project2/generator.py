import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 학습된 생성자 G (DCGAN). 수정하지 마세요.
#   z(128) -> 1024x4x4 -> 512x8x8 -> 256x16x16 -> 128x32x32 -> 3x64x64
# ============================================================
class Generator(nn.Module):
    def __init__(self, latent_dim=128, ch=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.deconv1 = nn.ConvTranspose2d(latent_dim, ch * 8, kernel_size=4, stride=1, padding=0)   # 1 -> 4
        self.bn1 = nn.BatchNorm2d(ch * 8)
        self.deconv2 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=4, stride=2, padding=1)       # 4 -> 8
        self.bn2 = nn.BatchNorm2d(ch * 4)
        self.deconv3 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1)       # 8 -> 16
        self.bn3 = nn.BatchNorm2d(ch * 2)
        self.deconv4 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=4, stride=2, padding=1)           # 16 -> 32
        self.bn4 = nn.BatchNorm2d(ch)
        self.deconv5 = nn.ConvTranspose2d(ch, 3, kernel_size=4, stride=2, padding=1)                # 32 -> 64

    def forward(self, z):                                    # (B, 128)
        x = z.view(-1, self.latent_dim, 1, 1)
        x = F.relu(self.bn1(self.deconv1(x)))                # (B, 1024, 4, 4)
        x = F.relu(self.bn2(self.deconv2(x)))                # (B, 512, 8, 8)
        x = F.relu(self.bn3(self.deconv3(x)))                # (B, 256, 16, 16)
        x = F.relu(self.bn4(self.deconv4(x)))                # (B, 128, 32, 32)
        return torch.tanh(self.deconv5(x))                   # (B, 3, 64, 64), 값 범위 [-1, 1]


def load_generator(path="G.pt"):
    G = Generator()
    G.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    G.eval()
    return G
