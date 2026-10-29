import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 공개 속성 분류기: 64x64 얼굴 -> 속성 20개 각각의 점수(logit)
#   속성 순서는 cond 와 같습니다. 확률은 torch.sigmoid(점수)
#   실제 얼굴 사진으로 학습됨. 채점 서버는 이것과 다른 비공개 분류기를 씁니다.
# ============================================================
class AttrClassifier(nn.Module):
    def __init__(self, n_attr=20, w=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, w, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(w)
        self.conv2 = nn.Conv2d(w, w * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(w * 2)
        self.conv3 = nn.Conv2d(w * 2, w * 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(w * 4)
        self.conv4 = nn.Conv2d(w * 4, w * 8, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(w * 8)
        self.fc = nn.Linear(w * 8, n_attr)

    def features(self, x):                                   # (B, 3, 64, 64), 값 범위 [-1, 1]
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)     # (B, 32, 32, 32)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)     # (B, 64, 16, 16)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)     # (B, 128, 8, 8)
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)     # (B, 256, 4, 4)
        return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)     # (B, 256) 얼굴의 특징 벡터

    def forward(self, x):
        return self.fc(self.features(x))                         # (B, 20)


def load_classifier(path="classifier.pt"):
    C = AttrClassifier()
    C.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    C.eval()
    return C
