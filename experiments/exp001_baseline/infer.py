# infer.py（全文）
"""
Inference script (single-file bundle)
- dataset.py / model.py を infer.py に内包
- metrics は不要（OOF 評価は run.py 側で行う）
- utils/env.py, utils/logger.py, utils/timing.py は外部のまま利用
"""
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

# プロジェクトルートを import パスに追加（infer.py が experiments/expXXX/... にある想定）
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
import timm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

try:
    from torch_ema import ExponentialMovingAverage
    _HAS_EMA = True
except Exception:
    _HAS_EMA = False

from utils.logger import get_logger
from utils.timing import trace


# =========================================================
# 1) Seed
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 2) Targets derivation（submission 用）
# =========================================================
def derive_secondary_from_primary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"Dry_Total_g", "GDM_g", "Dry_Green_g"}.issubset(df.columns):
        df["Dry_Dead_g"] = np.maximum(0.0, df["Dry_Total_g"].values - df["GDM_g"].values)
        df["Dry_Clover_g"] = np.maximum(0.0, df["GDM_g"].values - df["Dry_Green_g"].values)
    return df


# =========================================================
# 3) Dataset / Augment
# =========================================================
class BiomassTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        if "image_path" not in self.df.columns:
            raise KeyError(f"test.csv に 'image_path' 列がありません。df.columns={list(self.df.columns)}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = self.df.loc[idx, "image_path"]
        full_path = self.img_dir / img_path

        image = cv2.imread(str(full_path))
        if image is None:
            raise FileNotFoundError(f"画像が読めません: {full_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return {"image": image, "image_path": img_path}


def build_transforms(aug_cfg: dict, mode: str, default_size: int = 224):
    mode_cfg = (aug_cfg or {}).get(mode, {}) or {}

    tfs: List[A.BasicTransform] = []
    has_resize = False

    for name, params in mode_cfg.items():
        if name == "Resize":
            tfs.append(A.Resize(**params))
            has_resize = True
        elif name == "RandomResizedCrop":
            tfs.append(A.RandomResizedCrop(**params))
            has_resize = True
        elif name == "Normalize":
            tfs.append(A.Normalize(**params))

    if not has_resize:
        tfs.insert(0, A.Resize(default_size, default_size))

    tfs.append(ToTensorV2())
    return A.Compose(tfs)


# =========================================================
# 4) Model（Lightning / load_from_checkpoint 用に同一クラス定義が必要）
# =========================================================
class BiomassModel(pl.LightningModule):
    def __init__(
        self,
        backbone: str,
        num_targets: int,
        lr: float,
        weight_decay: float,
        warmup_ratio: float,
        loss_weights: Dict[str, float],
        ema_enabled: bool,
        ema_decay: float,
        total_steps: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_ratio = float(warmup_ratio)
        self.total_steps = total_steps

        self.loss_weights = torch.tensor(list(loss_weights.values()), dtype=torch.float32)

        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            feat_dim = 768

        self.head = nn.Sequential(
            nn.Linear(int(feat_dim), 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, int(num_targets)),
        )

        self.ema = None
        self.ema_enabled = bool(ema_enabled) and _HAS_EMA
        self.ema_decay = float(ema_decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)

    # 推論だけなら optimizer は不要だが、Lightning の互換性のため残す
    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.ema_enabled:
            self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        if self.total_steps is None:
            return opt
        warmup_steps = int(self.total_steps * self.warmup_ratio)
        sch = get_cosine_schedule_with_warmup(opt, warmup_steps, int(self.total_steps))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}


def find_checkpoint(fold_dir: Path, expected_stem: str) -> Path:
    """
    1) {expected_stem}.ckpt があればそれ
    2) なければ fold_dir 内の *.ckpt を名前順で最後のもの
    """
    p = fold_dir / f"{expected_stem}.ckpt"
    if p.exists():
        return p

    ckpts = sorted(fold_dir.glob("*.ckpt"))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"ckpt が見つかりません: {fold_dir}")
    return ckpts[-1]


def load_model_from_checkpoint(ckpt_path: Path, exp_cfg: dict, device: str):
    model = BiomassModel.load_from_checkpoint(
        str(ckpt_path),
        backbone=exp_cfg["model"]["backbone"],
        num_targets=int(exp_cfg["model"]["num_targets"]),
        lr=float(exp_cfg["optimizer"]["lr"]),
        weight_decay=float(exp_cfg["optimizer"]["weight_decay"]),
        warmup_ratio=float(exp_cfg["scheduler"]["warmup_ratio"]),
        loss_weights=exp_cfg["training"]["loss_weights"],
        ema_enabled=bool(exp_cfg["ema"]["enabled"]),
        ema_decay=float(exp_cfg["ema"]["decay"]),
    )
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_one_model(model, loader: DataLoader, device: str) -> np.ndarray:
    preds = []
    for batch in tqdm(loader, desc="infer", leave=False):
        x = batch["image"].to(device)
        p = model(x).detach().cpu().numpy()
        preds.append(p)
    return np.concatenate(preds, axis=0)


# =========================================================
# 5) main
# =========================================================
def main():
    exp_dir = Path(__file__).resolve().parent

    with open(exp_dir / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    with open(exp_dir / "exp" / "000.yaml", "r") as f:
        exp_cfg = yaml.safe_load(f)

    set_seed(int(exp_cfg.get("seed", 42)))

    artifacts_base = Path(cfg["output"]["artifacts_base"])
    exp_name = str(cfg["output"]["exp_name"])
    exp_id = str(exp_cfg["exp_id"])

    artifacts_dir = artifacts_base / exp_name / exp_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log_dir = artifacts_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(file_name=f"infer_{exp_name}_{exp_id}", file_dir=log_dir)
    logger.info(f"Artifacts dir: {artifacts_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    test_csv = Path(cfg["data"]["input_dir"]) / str(cfg["data"]["test_csv"])
    with trace("Loading test.csv"):
        test_df = pd.read_csv(test_csv)
        logger.info(f"Loaded test rows: {len(test_df)} from {test_csv}")

    tf = build_transforms(exp_cfg.get("augmentation", {}), mode="valid")
    test_ds = BiomassTestDataset(test_df, img_dir=str(Path(cfg["data"]["input_dir"])), transform=tf)

    test_loader = DataLoader(
        test_ds,
        batch_size=int(exp_cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(exp_cfg["training"]["num_workers"]),
        pin_memory=True,
    )

    n_folds = int(cfg["cv"]["n_folds"])
    ckpt_stem = str(exp_cfg["callbacks"]["model_checkpoint"]["filename"])

    all_fold_preds = []
    with trace("Loading checkpoints and inference"):
        for fold in range(n_folds):
            fold_dir = artifacts_dir / f"fold{fold}"
            if not fold_dir.exists():
                logger.warning(f"fold dir not found: {fold_dir}")
                continue

            ckpt_path = find_checkpoint(fold_dir, ckpt_stem)
            logger.info(f"fold{fold} ckpt: {ckpt_path.name}")

            model = load_model_from_checkpoint(ckpt_path, exp_cfg, device=device)
            pred = infer_one_model(model, test_loader, device=device)  # (N, 3)
            all_fold_preds.append(pred)

    if len(all_fold_preds) == 0:
        raise RuntimeError("推論に使える ckpt が見つかりません。artifacts_dir 配下を確認してください。")

    with trace("Ensemble"):
        ens = np.mean(np.stack(all_fold_preds, axis=0), axis=0)  # (N, 3)

    # primary prediction
    out = pd.DataFrame({
        "image_path": test_df["image_path"].values,
        "Dry_Total_g": ens[:, 0],
        "GDM_g": ens[:, 1],
        "Dry_Green_g": ens[:, 2],
    })
    out = derive_secondary_from_primary(out)

    # submission
    targets_all = cfg["targets"]["all"]
    rows = []
    for i in range(len(out)):
        img_path = out.loc[i, "image_path"]
        img_id = Path(str(img_path)).stem
        for t in targets_all:
            rows.append({"sample_id": f"{img_id}__{t}", "target": float(out.loc[i, t])})
    sub = pd.DataFrame(rows)

    sub_dir = artifacts_dir / "submission"
    sub_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    sub_path = sub_dir / f"SUB_{exp_name}_{exp_id}_{ts}.csv"
    sub.to_csv(sub_path, index=False)

    logger.info(f"Saved submission: {sub_path}")
    logger.info(f"submission shape: {sub.shape}")
    logger.info(sub.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
