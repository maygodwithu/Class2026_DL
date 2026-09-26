# DeepLearning Project 1: 식물 잎 사진으로 작물과 건강 상태 분류하기

2026년 2학기 딥러닝기초 **첫 번째 프로젝트**입니다.

식물 잎 사진 한 장을 보고 **두 가지를 동시에** 맞히는 CNN을 만듭니다.

| 과제 | 클래스 |
|---|---|
| 작물 종류 (`crop_label`) | 0: Apple, 1: Grape, 2: Tomato |
| 건강 상태 (`health_label`) | 0: Healthy, 1: Diseased |

하나의 CNN이 이미지에서 특징을 뽑고, 끝에 출력층을 두 개(작물용, 건강 상태용) 달아서 두 문제를 함께 풉니다.

---

## 1. 데이터

```
student_data_128/
├── class_mapping.json      라벨 번호 -> 이름
├── train.csv               id, crop_label, health_label
├── train/images/*.jpg      8,000장
├── valid.csv
└── valid/images/*.jpg      2,284장
```

- 모든 이미지는 **128 x 128 RGB JPEG**입니다.
- CSV 한 줄이 이미지 한 장입니다. 예: `train_010761.jpg,0,0` → Apple, Healthy
- **클래스 불균형이 큽니다.** (train 기준)

| | Healthy | Diseased | 합계 |
|---|---|---|---|
| Apple | 621 | 646 | 1,267 (16%) |
| Grape | 192 | 1,304 | 1,496 (19%) |
| Tomato | 581 | 4,656 | 5,237 (65%) |
| 합계 | 1,394 (17%) | 6,606 (83%) | 8,000 |

---

## 2. 코드 구성

| 파일 | 내용 |
|---|---|
| `dataset.py` | CSV와 이미지를 읽는 `PlantDataset`, 전처리/데이터 증강 `get_transforms` |
| `model.py` | CNN 모델 `MultiTaskCNN` (**여러분이 고칠 파일**) |
| `train.py` | 학습(`train`), 평가(`evaluate`), 결과 출력(`print_report`) |

---

## 3. 실행 방법

### Colab
```
!git clone https://github.com/maygodwithu/Class2026_DL.git
%cd Class2026_DL
!python train.py
```
런타임 유형을 GPU로 바꾸면 훨씬 빠릅니다.

### 내 컴퓨터
```
git clone https://github.com/maygodwithu/Class2026_DL.git
cd Class2026_DL
python train.py                  # GPU가 있으면 GPU, 없으면 CPU
python train.py --device cpu     # CPU로 실행
```
필요한 패키지: `torch`, `torchvision`, `Pillow`

### 자주 쓰는 옵션
```
python train.py --epochs 20 --lr 0.0005 --batch_size 64
python train.py --eval_only      # 저장된 best_model.pt 를 불러와 valid 로 평가만 하기
```

학습이 끝나면 valid 점수가 가장 좋았던 모델이 `best_model.pt`로 저장됩니다.

---

## 4. 과제

**`model.py`를 고쳐서 성능을 올리세요.**

제공된 `model.py`는 일부러 단순하게 만든 기본(baseline) 모델입니다.
층의 개수, 채널 수, 정규화, 규제 등 모델 구조를 바꿔 가며 성능을 개선해 보세요.

학습 설정(epoch 수, 학습률, 데이터 증강 등)은 `train.py`, `dataset.py`에서 자유롭게 바꿔도 됩니다.

---

## 5. 제출과 채점

1. 학습이 끝나면 **`model.py`와 `best_model.pt`** 두 파일을 리더보드에 제출합니다.
2. 서버가 공개되지 않은 **test 세트**로 채점해서 등수를 보여 줍니다.
3. 리더보드 주소는 수업 시간에 안내합니다.

### 채점 지표: 평균 Macro-F1
```
score = (작물 종류 Macro-F1 + 건강 상태 Macro-F1) / 2
```
- `train.py`가 출력하는 `score`와 같은 값입니다.
- 건강 상태는 83%가 Diseased라서, 전부 Diseased로 찍어도 Accuracy가 83%나 나옵니다.
  그래서 Accuracy 대신 **클래스마다 고르게 잘 맞혀야 높게 나오는 Macro-F1**로 채점합니다.

### 서버에서 채점되려면 꼭 지켜야 할 것
서버는 여러분의 `model.py`에서 모델을 만들고 `best_model.pt`를 불러와 채점합니다.
아래를 지키지 않으면 채점이 되지 않습니다.

- 모델 클래스 이름은 **`MultiTaskCNN`**
- **`MultiTaskCNN()`처럼 인자 없이** 모델을 만들 수 있어야 함
- 입력: `(B, 3, 128, 128)` 이미지
- 출력: **`(crop_logits, health_logits)`** 두 개를 반환, 각각 `(B, 3)`, `(B, 2)`
- 평가할 때 전처리는 `dataset.py`의 `get_transforms(train=False)` (ToTensor + Normalize(0.5, 0.5))를 그대로 사용

---

## 6. 시상

| 등수 | 혜택 |
|---|---|
| 1 ~ 3등 | **상품** |
| 1 ~ 20등 | **가산점** |
