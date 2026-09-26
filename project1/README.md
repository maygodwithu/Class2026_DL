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
%cd Class2026_DL/project1
!python train.py
```
런타임 유형을 GPU로 바꾸면 훨씬 빠릅니다.

### 내 컴퓨터
```
git clone https://github.com/maygodwithu/Class2026_DL.git
cd Class2026_DL/project1
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

### 제한 사항
| 항목 | 제한 |
|---|---|
| **사전학습(pretrained) 모델** | **사용 금지.** ImageNet 등으로 미리 학습된 가중치를 가져오지 말고, 주어진 데이터로 **처음부터 직접 학습**하세요 |
| **모델 크기** | 파라미터 **2,000,000개(2M) 이하** |
| **`best_model.pt` 파일 크기** | **10MB 이하** |
| **사용 가능한 라이브러리** | `torch`, `torchvision`만 (채점 서버에 이 둘만 설치되어 있음) |

- 파라미터 수는 `train.py`를 실행하면 처음에 `파라미터 수: ...`로 출력됩니다. 제출 전에 확인하세요.
  (채점 서버는 BatchNorm 통계 같은 버퍼도 함께 세기 때문에 조금 더 크게 셀 수 있습니다. 여유를 두세요.)
- 참고: 제공된 baseline은 약 2.4만 개, 4블록 + BatchNorm 정도의 CNN은 약 40만 개입니다.

---

## 5. 제출과 채점

### 리더보드: http://203.253.70.211:8814

| | |
|---|---|
| **마감** | **2026년 10월 13일 (화) 23:59** (한국 시간). 마감 후에는 제출할 수 없습니다 |
| **제출 횟수** | 학번당 **하루 10회** (한국 시간 기준, 채점에 실패한 제출도 포함) |
| **제출물** | `model.py` + `best_model.pt` |

1. 리더보드의 **[제출]** 페이지에서 학번(필수), 이름 또는 닉네임(선택), 두 파일을 올립니다.
2. 서버가 공개되지 않은 **test 세트**로 자동 채점합니다(보통 수십 초). 채점이 끝나면 내 제출 페이지에 점수가 나타납니다.
3. 채점에 실패하면 이유가 표시됩니다. 아래 "서버에서 채점되려면 꼭 지켜야 할 것"을 확인하세요.
4. **[순위]** 페이지에서 전체 순위를 볼 수 있습니다.

### Public / Private 점수
test 세트는 **public**(절반)과 **private**(나머지 절반)으로 나뉘어 있습니다.

- 기간 중에는 **public 점수**만 보이고, 순위도 public 점수로 보여 줍니다.
- 학생마다 **public 점수가 가장 높은 제출 1개**가 최종 제출이 됩니다.
- **최종 순위는 마감 후 그 제출의 private 점수**로 정합니다. 마감 후 리더보드가 private 순위로 바뀝니다.
- 그래서 public 점수만 보고 운 좋게 맞춘 모델보다, **처음 보는 데이터에서도 잘 맞히는 모델**이 최종적으로 유리합니다.
  public 점수를 올리려고 무작정 여러 번 제출하기보다 valid 점수로 충분히 검증한 뒤 제출하세요.

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
- 파라미터 2M 이하, `best_model.pt` 10MB 이하, `torch`/`torchvision` 외 라이브러리 사용 불가 (위 "제한 사항" 참고)
- `best_model.pt`는 `torch.save(model.state_dict(), ...)`로 저장한 파일 (`train.py`가 저장하는 그대로)

---

## 6. 시상

| 등수 | 혜택 |
|---|---|
| 1 ~ 3등 | **상품** |
| 1 ~ 20등 | **가산점** |
