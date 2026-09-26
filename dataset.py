import os
import csv
import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_class_names(data_dir):
    """class_mapping.json -> (작물 이름 리스트, 건강 상태 이름 리스트)"""
    with open(os.path.join(data_dir, "class_mapping.json"), encoding="utf-8") as f:
        m = json.load(f)
    crop_names = [m["crop_label"][str(i)] for i in range(len(m["crop_label"]))]
    health_names = [m["health_label"][str(i)] for i in range(len(m["health_label"]))]
    return crop_names, health_names


class PlantDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.img_dir = os.path.join(data_dir, split, "images")
        self.transform = transform
        # train.csv 앞에 BOM이 있으므로 utf-8-sig 로 읽는다
        with open(os.path.join(data_dir, f"{split}.csv"), encoding="utf-8-sig") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(os.path.join(self.img_dir, row["id"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        crop = int(row["crop_label"])
        health = int(row["health_label"])
        return image, crop, health


def get_transforms(train):
    """이미지는 이미 128x128 로 맞춰져 있으므로 Resize 는 하지 않는다."""
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if train:
        # 학습 시에만 간단한 데이터 증강 (좌우/상하 뒤집기)
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
