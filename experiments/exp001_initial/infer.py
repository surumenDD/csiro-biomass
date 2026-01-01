# experiments/exp001_initial/infer.py

import os
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
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
OUTPUT_DIR = ROOT / "output"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_NAMES = [
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "GDM_g",
    "Dry_Total_g",
]


# =====================
# main
# =====================
def main():
    print("PROJECT_ROOT:", ROOT)

    test_df = pd.read_csv(INPUT_DIR / "test.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model = timm.create_model(
        "resnet18",
        pretrained=False,
        num_classes=5,
    )

    model.load_state_dict(
        torch.load(OUTPUT_DIR / "model.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    results = []

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            img_name = Path(row["image_path"]).name
            img = Image.open(INPUT_DIR / "test" / img_name).convert("RGB")
            img = transform(img).unsqueeze(0).to(DEVICE)

            preds = model(img).cpu().numpy()[0]
            pred_dict = dict(zip(TARGET_NAMES, preds))

            results.append({
                "sample_id": row["sample_id"],
                "target": float(pred_dict[row["target_name"]]),
            })

    out_path = OUTPUT_DIR / "submission.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    print("saved:", out_path)


if __name__ == "__main__":
    main()
