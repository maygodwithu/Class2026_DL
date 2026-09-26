import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PlantDataset, get_transforms, load_class_names
from model import MultiTaskCNN


# ============================================================
# 평가 지표
# ============================================================
def accuracy(conf):
    """혼동행렬 -> 정확도"""
    return (conf.diag().sum() / conf.sum()).item()


def macro_f1(conf):
    """혼동행렬(conf[정답, 예측]) -> (클래스별 F1 의 평균, 클래스별 F1 리스트)"""
    tp = conf.diag().float()
    precision = tp / conf.sum(0).clamp(min=1)
    recall = tp / conf.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return f1.mean().item(), f1.tolist()


# ============================================================
# 평가
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """모델을 loader 의 데이터로 평가한다.

    반환: dict
      loss        : 평균 손실 (criterion 을 줬을 때만)
      crop_acc, crop_f1, crop_f1_per_class, crop_conf
      health_acc, health_f1, health_f1_per_class, health_conf
      score       : 두 과제 Macro-F1 의 평균 (최고 모델 선택 기준)
    """
    model.eval()
    num_crops = model.crop_head.out_features
    num_health = model.health_head.out_features
    crop_conf = torch.zeros(num_crops, num_crops, dtype=torch.long)
    health_conf = torch.zeros(num_health, num_health, dtype=torch.long)
    total_loss, n = 0.0, 0

    for images, crops, healths in loader:
        images, crops, healths = images.to(device), crops.to(device), healths.to(device)
        crop_out, health_out = model(images)

        if criterion is not None:
            loss = criterion(crop_out, crops) + criterion(health_out, healths)
            total_loss += loss.item() * images.size(0)
        n += images.size(0)

        # 혼동행렬 누적 (행: 정답, 열: 예측)
        for t, p in zip(crops.cpu(), crop_out.argmax(1).cpu()):
            crop_conf[t, p] += 1
        for t, p in zip(healths.cpu(), health_out.argmax(1).cpu()):
            health_conf[t, p] += 1

    crop_f1, crop_f1_per_class = macro_f1(crop_conf)
    health_f1, health_f1_per_class = macro_f1(health_conf)
    return {
        "loss": total_loss / n if criterion is not None else None,
        "crop_acc": accuracy(crop_conf),
        "crop_f1": crop_f1,
        "crop_f1_per_class": crop_f1_per_class,
        "crop_conf": crop_conf,
        "health_acc": accuracy(health_conf),
        "health_f1": health_f1,
        "health_f1_per_class": health_f1_per_class,
        "health_conf": health_conf,
        "score": (crop_f1 + health_f1) / 2,
    }


def print_report(result, crop_names, health_names):
    """evaluate() 결과를 보기 좋게 출력한다."""
    for title, key, names in [("작물 종류", "crop", crop_names),
                              ("건강 상태", "health", health_names)]:
        print(f"\n[{title}] Accuracy {result[key + '_acc']:.3f}, Macro-F1 {result[key + '_f1']:.3f}")
        for name, v in zip(names, result[key + "_f1_per_class"]):
            print(f"  {name:10s} F1 {v:.3f}")
        print("  혼동행렬 (행: 정답, 열: 예측)")
        print("  " + str(result[key + "_conf"].tolist()))
    print(f"\n평균 Macro-F1 (score): {result['score']:.3f}")


# ============================================================
# 학습 (전체): epochs 만큼 학습하면서 매 epoch 평가, 최고 모델 저장
# ============================================================
def train(model, train_loader, eval_loader, criterion, optimizer, device, epochs, save_path):
    """반환: epoch 별 기록 리스트 (학습 곡선을 그릴 때 사용)"""
    history = []
    best_score = -1.0
    for epoch in range(1, epochs + 1):
        start = time.time()

        # ----- 1 epoch 학습 -----
        model.train()
        total_loss, n = 0.0, 0
        for images, crops, healths in train_loader:
            images, crops, healths = images.to(device), crops.to(device), healths.to(device)

            crop_out, health_out = model(images)
            loss = criterion(crop_out, crops) + criterion(health_out, healths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            n += images.size(0)
        train_loss = total_loss / n

        # ----- 평가 -----
        r = evaluate(model, eval_loader, device, criterion)

        print(f"[{epoch:2d}/{epochs}] "
              f"train_loss {train_loss:.4f} | valid_loss {r['loss']:.4f} | "
              f"crop acc {r['crop_acc']:.3f} F1 {r['crop_f1']:.3f} | "
              f"health acc {r['health_acc']:.3f} F1 {r['health_f1']:.3f} | "
              f"{time.time() - start:.0f}s")

        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": r["loss"],
                        "crop_f1": r["crop_f1"], "health_f1": r["health_f1"], "score": r["score"]})

        if r["score"] > best_score:
            best_score = r["score"]
            torch.save(model.state_dict(), save_path)
            print(f"  -> 최고 모델 저장 (평균 Macro-F1 {best_score:.3f})")
    return history


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="student_data_128")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto: GPU 가 있으면 GPU, 없으면 CPU")
    parser.add_argument("--save_path", default="best_model.pt")
    parser.add_argument("--eval_only", action="store_true",
                        help="학습하지 않고 save_path 의 모델을 불러와 평가만 한다")
    parser.add_argument("--eval_split", default="valid",
                        help="평가할 데이터 (data_dir 안의 <split>.csv, <split>/images)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"device: {device}")

    torch.manual_seed(42)

    # ---------------- 데이터 / 모델 ----------------
    crop_names, health_names = load_class_names(args.data_dir)
    eval_set = PlantDataset(args.data_dir, args.eval_split, get_transforms(train=False))
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    model = MultiTaskCNN(num_crops=len(crop_names), num_health=len(health_names)).to(device)

    # ---------------- 평가만 하기 ----------------
    if args.eval_only:
        model.load_state_dict(torch.load(args.save_path, map_location=device))
        print(f"'{args.save_path}' 모델로 {args.eval_split} ({len(eval_set)}장) 평가")
        print_report(evaluate(model, eval_loader, device), crop_names, health_names)
        return

    # ---------------- 학습 ----------------
    train_set = PlantDataset(args.data_dir, "train", get_transforms(train=True))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    print(f"train: {len(train_set)}장, {args.eval_split}: {len(eval_set)}장")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    train(model, train_loader, eval_loader, criterion, optimizer, device,
          args.epochs, args.save_path)

    # ---------------- 최고 모델로 최종 평가 ----------------
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    print(f"\n===== 최종 결과 ({args.eval_split}) =====")
    print_report(evaluate(model, eval_loader, device), crop_names, health_names)


if __name__ == "__main__":
    main()
