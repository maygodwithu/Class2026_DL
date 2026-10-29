import argparse

import numpy as np
import torch
from torchvision.utils import save_image

from generator import load_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="targets200.npy", help="목표 사진 (N, 64, 64, 3) uint8")
    parser.add_argument("--n_random", type=int, default=1000, help="랜덤으로 그려 볼 얼굴 수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="z.pt", help="제출할 z 파일")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    G = load_generator("G.pt")

    # 목표 사진: (N, 64, 64, 3) 0~255  ->  (N, 3, 64, 64) [-1, 1]  (G 의 출력과 같은 형식)
    target = torch.from_numpy(np.load(args.targets)).permute(0, 3, 1, 2).float() / 127.5 - 1
    print(f"목표 사진 {len(target)}장")

    # 1) 랜덤 z 를 n_random 개 뽑아 얼굴을 그린다
    z_random = torch.randn(args.n_random, 128)
    with torch.no_grad():
        faces = torch.cat([G(z_random[i:i + 100]) for i in range(0, args.n_random, 100)])

    # 2) 목표 사진마다 픽셀 차이가 가장 작은 얼굴을 고른다
    dist = torch.cdist(target.flatten(1), faces.flatten(1))         # (N, n_random) 사진과 얼굴 사이 거리
    best = dist.argmin(dim=1)                                        # (N,) 가장 가까운 얼굴의 번호
    z = z_random[best]                                               # (N, 128) 제출할 z

    # 3) z 를 파일로 저장 + 확인용 그림 (홀수 줄: 목표 사진, 짝수 줄: 찾은 z 로 그린 얼굴)
    torch.save(z, args.out)
    found = faces[best]
    mse = ((found - target) ** 2).mean().item()
    rows = []
    for i in range(0, 40, 10):
        rows += [target[i:i + 10], found[i:i + 10]]
    save_image((torch.cat(rows) + 1) / 2, "compare.png", nrow=10)
    print(f"z 저장: {args.out}  {tuple(z.shape)}")
    print(f"평균 픽셀 오차(MSE): {mse:.4f}  (작을수록 비슷함)")
    print("확인용 그림: compare.png (앞 40장)")


if __name__ == "__main__":
    main()
