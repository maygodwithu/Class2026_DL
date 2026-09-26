"""제출 전 점검: 채점 서버와 같은 방식으로 model.py + best_model.pt 를 검사한다.

사용법 (project1 폴더에서)
  python check_submission.py
  python check_submission.py --model model.py --weights best_model.pt --data_dir student_data_128

모든 항목이 [통과] 이면 리더보드에 제출해도 채점됩니다.
채점 서버처럼 CPU 로만 실행합니다.
"""
import os
import ast
import sys
import csv
import argparse
import importlib.util

import torch
from PIL import Image
from torchvision import transforms

# 채점 서버와 같은 제한
MAX_MODEL_BYTES = 200 * 1024
MAX_WEIGHT_BYTES = 10 * 1024 * 1024
MAX_PARAMS = 2_000_000
ALLOWED_PACKAGES = {"torch", "torchvision"}
NUM_CROPS, NUM_HEALTH = 3, 2

# 채점 서버가 쓰는 평가 전처리 (dataset.py 의 get_transforms(train=False) 와 같음)
EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def ok(msg):
    print(f"[통과] {msg}")


def fail(msg):
    print(f"[실패] {msg}")
    print("\n=> 위 문제를 고친 뒤 다시 실행하세요. 이대로 제출하면 채점에 실패합니다.")
    sys.exit(1)


def check_imports(path):
    """torch, torchvision, 파이썬 기본 라이브러리 외의 import 가 있는지 검사"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return sorted(n for n in names if n not in ALLOWED_PACKAGES and n not in sys.stdlib_module_names)


def macro_f1(pairs, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for t, p in pairs if t == c and p == c)
        fp = sum(1 for t, p in pairs if t != c and p == c)
        fn = sum(1 for t, p in pairs if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / num_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.py")
    parser.add_argument("--weights", default="best_model.pt")
    parser.add_argument("--data_dir", default="student_data_128")
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    print("=" * 60)
    print(" 제출 전 점검 (채점 서버와 같은 방식, CPU)")
    print("=" * 60)

    # 1. 파일
    for path, limit, name in [(args.model, MAX_MODEL_BYTES, "model.py"), (args.weights, MAX_WEIGHT_BYTES, "best_model.pt")]:
        if not os.path.isfile(path):
            fail(f"{path} 파일이 없습니다")
        size = os.path.getsize(path)
        if size > limit:
            fail(f"{name} 파일이 너무 큽니다: {size / 1024 / 1024:.2f}MB (제한 {limit / 1024 / 1024:.2f}MB)")
    ok(f"파일 크기: model.py {os.path.getsize(args.model) / 1024:.1f}KB, "
       f"best_model.pt {os.path.getsize(args.weights) / 1024 / 1024:.2f}MB (제한 10MB)")

    # 2. import
    bad = check_imports(args.model)
    if bad:
        fail(f"채점 서버에 없는 라이브러리를 import 합니다: {', '.join(bad)} (torch, torchvision 만 사용 가능)")
    ok("import: torch, torchvision, 파이썬 기본 라이브러리만 사용")

    # 3. model.py 불러오기
    spec = importlib.util.spec_from_file_location("student_model", args.model)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        fail(f"model.py 를 불러오지 못했습니다 ({type(e).__name__}: {e})")
    if not hasattr(module, "MultiTaskCNN"):
        fail("model.py 에 MultiTaskCNN 클래스가 없습니다")
    ok("model.py 에 MultiTaskCNN 클래스가 있음")

    # 4. 인자 없이 모델 만들기
    try:
        model = module.MultiTaskCNN()
    except Exception as e:
        fail(f"MultiTaskCNN() 으로 모델을 만들지 못했습니다. __init__ 의 모든 인자에 기본값이 있어야 합니다 "
             f"({type(e).__name__}: {e})")
    ok("MultiTaskCNN() 을 인자 없이 만들 수 있음")

    # 5. 모델 크기 (학습 파라미터 + 실수형 버퍼)
    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers() if b.is_floating_point())
    total = n_params + n_buffers
    if total > MAX_PARAMS:
        fail(f"모델이 너무 큽니다: 파라미터 {n_params:,}개 + 버퍼 {n_buffers:,}개 = {total:,}개 (제한 {MAX_PARAMS:,}개)")
    ok(f"모델 크기: 파라미터 {n_params:,}개 + 버퍼 {n_buffers:,}개 = {total:,}개 (제한 {MAX_PARAMS:,}개)")

    # 6. 가중치 불러오기
    try:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
    except Exception as e:
        fail(f"best_model.pt 를 읽지 못했습니다. torch.save(model.state_dict(), ...) 로 저장했는지 확인하세요 "
             f"({type(e).__name__}: {e})")
    try:
        model.load_state_dict(state)
    except Exception as e:
        fail(f"best_model.pt 가 model.py 의 구조와 맞지 않습니다. 학습할 때 쓴 model.py 와 같은지 확인하세요 "
             f"({type(e).__name__}: {str(e)[:300]})")
    model.eval()
    ok("best_model.pt 를 model.py 구조에 맞게 불러옴")

    # 7. valid 전체 예측 (출력 형식 검사 + 점수)
    csv_path = os.path.join(args.data_dir, "valid.csv")
    if not os.path.isfile(csv_path):
        fail(f"{csv_path} 가 없습니다. --data_dir 로 데이터 폴더를 지정하세요")
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    crop_pairs, health_pairs = [], []
    with torch.no_grad():
        for i in range(0, len(rows), 128):
            batch = rows[i:i + 128]
            x = torch.stack([EVAL_TRANSFORM(Image.open(os.path.join(args.data_dir, "valid", "images", r["id"])).convert("RGB"))
                             for r in batch])
            try:
                out = model(x)
            except Exception as e:
                fail(f"모델 forward 중 오류 ({type(e).__name__}: {e})")
            if not (isinstance(out, (tuple, list)) and len(out) == 2):
                fail("모델 출력은 (crop_logits, health_logits) 두 개여야 합니다")
            crop_out, health_out = out
            n = len(batch)
            if tuple(crop_out.shape) != (n, NUM_CROPS) or tuple(health_out.shape) != (n, NUM_HEALTH):
                fail(f"출력 크기가 올바르지 않습니다: crop {tuple(crop_out.shape)} (기대 {(n, NUM_CROPS)}), "
                     f"health {tuple(health_out.shape)} (기대 {(n, NUM_HEALTH)})")
            if not (torch.isfinite(crop_out).all() and torch.isfinite(health_out).all()):
                fail("모델 출력에 NaN 또는 inf 가 있습니다")
            crop_pairs += zip([int(r["crop_label"]) for r in batch], crop_out.argmax(1).tolist())
            health_pairs += zip([int(r["health_label"]) for r in batch], health_out.argmax(1).tolist())
    ok(f"출력 형식: (B, 3), (B, 2)  (valid {len(rows)}장 예측 완료)")

    crop_f1 = macro_f1(crop_pairs, NUM_CROPS)
    health_f1 = macro_f1(health_pairs, NUM_HEALTH)
    print("-" * 60)
    print(f" valid 점수 (채점 서버와 같은 전처리)")
    print(f"   작물 종류 Macro-F1 : {crop_f1:.4f}")
    print(f"   건강 상태 Macro-F1 : {health_f1:.4f}")
    print(f"   평균 Macro-F1      : {(crop_f1 + health_f1) / 2:.4f}")
    print("=" * 60)
    print(" 모든 검사 통과! 리더보드에 model.py 와 best_model.pt 를 제출하세요.")
    print(" (test 점수는 valid 점수와 조금 다를 수 있습니다)")
    print("=" * 60)


if __name__ == "__main__":
    main()
