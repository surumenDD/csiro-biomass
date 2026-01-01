# experiments/exp001_initial/run.py

import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
from torchvision import transforms
from tqdm import tqdm


# =====================
# パス・設定
# =====================
def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / "Makefile").exists():
            return p
    raise RuntimeError(f"project root not found from: {start}")

try:
    HERE = Path(__file__).resolve().parent  # .py のとき
except NameError:
    HERE = Path.cwd().resolve()             # .ipynb のとき（現在作業ディレクトリ）

ROOT = find_project_root(HERE)

INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"  # ゆくゆくは artifacts に変更

if OUTPUT_DIR.exists() and not OUTPUT_DIR.is_dir():
    raise RuntimeError(f"{OUTPUT_DIR} exists but is not a directory")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 1
LR = 1e-4

TARGET_NAMES = [
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "GDM_g",
    "Dry_Total_g",
]


# =====================
# Dataset
# =====================
class BiomassDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        grouped = defaultdict(dict)

        for _, r in df.iterrows():
            image_id = r["sample_id"].split("__")[0]
            grouped[image_id]["image_path"] = r["image_path"]
            grouped[image_id][r["target_name"]] = r["target"]

        self.items = []
        for image_id, data in grouped.items():
            if not all(t in data for t in TARGET_NAMES):
                continue

            targets = torch.tensor(
                [data[t] for t in TARGET_NAMES],
                dtype=torch.float32
            )
            self.items.append((data["image_path"], targets))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, target = self.items[idx]
        img_name = Path(img_path).name

        img = Image.open(self.img_dir / img_name).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, target


# =====================
# main
# =====================
def main():
    print("PROJECT_ROOT:", ROOT)
    print("INPUT_DIR   :", INPUT_DIR)
    print("OUTPUT_DIR  :", OUTPUT_DIR)

    train_df = pd.read_csv(INPUT_DIR / "train.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = BiomassDataset(
        train_df,
        INPUT_DIR / "train",
        transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = timm.create_model(
        "resnet18",
        pretrained=True,
        num_classes=5,
    )
    model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}")
        for imgs, targets in pbar:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss))

    out_path = OUTPUT_DIR / "model.pth"
    torch.save(model.state_dict(), out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
