import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from generator import load_generator
from classifier import load_classifier

NORM_MIN, NORM_MAX = 8.5, 14.1          # z 크기(노름)의 정상 범위. 표준정규분포 z 128개는 거의 모두 이 안에 있다
TOL = 1e-3                              # 계산 오차 허용


def check(z, n):
    # 제출 파일 점검. 문제가 있으면 이유를 출력하고 False
    if not isinstance(z, torch.Tensor):
        print("[실패] z 파일에 텐서가 들어 있지 않습니다")
        return False
    if tuple(z.shape) != (n, 128):
        print(f"[실패] z 모양이 {tuple(z.shape)} 입니다. ({n}, 128) 이어야 합니다")
        return False
    print(f"[통과] z 모양 ({n}, 128)")
    if not torch.isfinite(z).all():
        print("[실패] z 에 NaN 또는 inf 가 있습니다")
        return False
    print("[통과] NaN / inf 없음")
    norm = z.norm(dim=1)
    bad = ((norm < NORM_MIN - TOL) | (norm > NORM_MAX + TOL)).sum().item()
    if bad > 0:
        print(f"[주의] 평균 0, 표준편차 1 범위를 크게 벗어난 z 가 {bad}개 있습니다. 그 사진은 0점입니다")
    else:
        print(f"[통과] z 범위: {n}개 모두 평균 0, 표준편차 1 근처")
    return True


@torch.no_grad()
def reference(G, C, target):
    # 기준: 사진마다 랜덤 얼굴 1000장과의 평균 차이 (랜덤 수준 = 0점)
    g = torch.Generator().manual_seed(999)
    ref = torch.cat([G(torch.randn(100, 128, generator=g)) for _ in range(10)])            # (1000, 3, 64, 64)
    f_t = F.normalize(C.features(target), dim=1)
    f_ref = F.normalize(C.features(ref), dim=1)
    mse_rand = torch.cdist(target.flatten(1), ref.flatten(1)).pow(2).mean(1) / target[0].numel()
    cos_rand = (f_t @ f_ref.T).mean(1)
    return f_t, mse_rand, cos_rand


@torch.no_grad()
def score_z(G, C, target, z, ref):
    # 사진마다 점수 (0 ~ 1). 반환: 점수, 픽셀 점수, 특징 점수, G(z)
    f_t, mse_rand, cos_rand = ref
    x = torch.cat([G(z[i:i + 100]) for i in range(0, len(z), 100)])
    mse = ((x - target) ** 2).mean(dim=(1, 2, 3))
    cos = (F.normalize(C.features(x), dim=1) * f_t).sum(1)
    pixel = (1 - mse / mse_rand).clamp(0, 1)                         # 색과 형태가 얼마나 같은가
    feature = ((cos - cos_rand) / (1 - cos_rand)).clamp(0, 1)        # 같은 사람처럼 보이는가
    norm = z.norm(dim=1)
    valid = ((norm >= NORM_MIN - TOL) & (norm <= NORM_MAX + TOL)).float()
    return (pixel + feature) / 2 * valid, pixel, feature, x


def load_targets(path):
    return torch.from_numpy(np.load(path)).permute(0, 3, 1, 2).float() / 127.5 - 1      # (N, 3, 64, 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z", default="z.pt", help="제출할 z 파일")
    parser.add_argument("--targets", default="targets200.npy")
    args = parser.parse_args()

    G = load_generator("G.pt")
    C = load_classifier("classifier.pt")
    target = load_targets(args.targets)
    z = torch.load(args.z, map_location="cpu").float()
    if not check(z, len(target)):
        return

    score, pixel, feature, x = score_z(G, C, target, z, reference(G, C, target))
    print("-" * 60)
    print(f"픽셀 점수 : {pixel.mean():.3f}")
    print(f"특징 점수 : {feature.mean():.3f}  (공개 분류기 기준)")
    print(f"참고 점수 : {score.mean() * 100:.1f} / 100  (서버 점수와 다를 수 있습니다)")

    worst = score.argsort()[:10]
    save_image((torch.cat([target[worst], x[worst]]) + 1) / 2, "worst.png", nrow=10)
    print(f"가장 안 맞은 사진 10장: {worst.tolist()}  -> worst.png (위: 사진, 아래: 내 결과)")


if __name__ == "__main__":
    main()
