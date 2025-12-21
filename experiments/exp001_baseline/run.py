# run.py（全文）
"""
Training script (single-file bundle)
- dataset.py / model.py / utils/cv.py / utils/metrics.py を run.py に内包
- utils/env.py, utils/logger.py, utils/timing.py は外部のまま利用
"""
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# プロジェクトルートを import パスに追加（run.py が experiments/expXXX/... にある想定）
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

try:
    from pytorch_lightning.loggers import WandbLogger
    _HAS_WANDB_LOGGER = True
except Exception:
    _HAS_WANDB_LOGGER = False

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

import timm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

try:
    from torch_ema import ExponentialMovingAverage
    _HAS_EMA = True
except Exception:
    _HAS_EMA = False

from sklearn.model_selection import StratifiedKFold

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 2) Targets derivation
# =========================================================
def derive_secondary_from_primary(df: pd.DataFrame) -> pd.DataFrame:
    """
    df は以下の列を持つ想定
    - Dry_Total_g
    - GDM_g
    - Dry_Green_g

    追加で以下を作る
    - Dry_Dead_g   = max(0, Dry_Total_g - GDM_g)
    - Dry_Clover_g = max(0, GDM_g - Dry_Green_g)
    """
    df = df.copy()
    if {"Dry_Total_g", "GDM_g", "Dry_Green_g"}.issubset(df.columns):
        df["Dry_Dead_g"] = np.maximum(0.0, df["Dry_Total_g"].values - df["GDM_g"].values)
        df["Dry_Clover_g"] = np.maximum(0.0, df["GDM_g"].values - df["Dry_Green_g"].values)
    return df


# =========================================================
# 3) Metrics (utils/metrics.py を削除する前提で内包)
# =========================================================
TARGET_WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}


def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    weighted_mean = np.sum(weights * y_true) / np.sum(weights)
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true - weighted_mean) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def calculate_oof_metrics(oof_df: pd.DataFrame) -> Dict[str, float]:
    """
    oof_df columns:
      - target_name
      - target
      - pred
    （他の列があっても良い）
    """
    all_y_true: List[float] = []
    all_y_pred: List[float] = []
    all_weights: List[float] = []
    per_target_r2: Dict[str, float] = {}

    for target_name in TARGET_WEIGHTS.keys():
        tdf = oof_df[oof_df["target_name"] == target_name]
        if len(tdf) == 0:
            continue
        y_true = tdf["target"].values.astype(np.float64)
        y_pred = tdf["pred"].values.astype(np.float64)
        w = float(TARGET_WEIGHTS[target_name])

        per_target_r2[target_name] = weighted_r2_score(y_true, y_pred, np.ones(len(y_true)) * w)
        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_weights.extend([w] * len(y_true))

    global_r2 = weighted_r2_score(np.array(all_y_true), np.array(all_y_pred), np.array(all_weights))

    return {
        "global_weighted_r2": float(global_r2),
        **{f"r2_{k}": float(v) for k, v in per_target_r2.items()},
    }


# =========================================================
# 4) Data normalize helpers（train.csv が long/wide どちらでも耐える）
# =========================================================
def _require_column(df: pd.DataFrame, col: str, name: str) -> None:
    if col not in df.columns:
        raise KeyError(f"{name} に必要な列 '{col}' がありません。df.columns={list(df.columns)}")


def to_wide_train_df(
    raw_df: pd.DataFrame,
    target_primary: List[str],
    group_col: Optional[str],
    stratify_col: Optional[str],
) -> pd.DataFrame:
    """
    raw_df が以下のどちらでも、学習用の wide（1行=1画像）に変換する。

    (A) long 形式: image_path, target_name, target (+ group/stratify 列)
    (B) wide 形式: image_path, Dry_Total_g, GDM_g, Dry_Green_g (+ group/stratify 列)
    """
    _require_column(raw_df, "image_path", "train.csv")

    # long 形式
    if {"target_name", "target"}.issubset(raw_df.columns):
        df = raw_df.copy()
        need = set(target_primary)
        df = df[df["target_name"].isin(list(need))].copy()

        wide = df.pivot_table(
            index="image_path",
            columns="target_name",
            values="target",
            aggfunc="mean",
        ).reset_index()

        # group/stratify を image_path 単位で付与
        if group_col and group_col in raw_df.columns:
            g = raw_df.groupby("image_path")[group_col].first().reset_index()
            wide = wide.merge(g, on="image_path", how="left")
        if stratify_col and stratify_col in raw_df.columns:
            s = raw_df.groupby("image_path")[stratify_col].first().reset_index()
            wide = wide.merge(s, on="image_path", how="left")

        # 欠損補完（基本は発生しない想定だが保険）
        for t in target_primary:
            if t not in wide.columns:
                wide[t] = 0.0
        return wide

    # wide 形式
    else:
        df = raw_df.copy()
        for t in target_primary:
            _require_column(df, t, "train.csv (wide形式)")
        keep_cols = ["image_path"] + target_primary
        if group_col and group_col in df.columns:
            keep_cols.append(group_col)
        if stratify_col and stratify_col in df.columns:
            keep_cols.append(stratify_col)
        return df[keep_cols].copy()


# =========================================================
# 5) CV split（utils/cv.py を削除する前提で内包）
# =========================================================
def stratified_group_kfold(
    df_wide: pd.DataFrame,
    n_folds: int,
    group_col: str,
    stratify_col: str,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    group_col を同一 fold に固定しつつ、stratify_col の分布を近づける簡易版。
    group_col / stratify_col が存在しない場合は StratifiedKFold にフォールバックする。
    """
    if group_col not in df_wide.columns or stratify_col not in df_wide.columns:
        # フォールバック
        if stratify_col not in df_wide.columns:
            raise KeyError(f"CV 分割に必要な '{stratify_col}' がありません。")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = []
        for tr, va in skf.split(df_wide, df_wide[stratify_col]):
            splits.append((tr, va))
        return splits

    group_df = df_wide.groupby(group_col)[stratify_col].agg(lambda x: x.mode()[0]).reset_index()
    group_df.columns = [group_col, stratify_col]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for tr_g, va_g in skf.split(group_df, group_df[stratify_col]):
        tr_groups = set(group_df.iloc[tr_g][group_col].tolist())
        va_groups = set(group_df.iloc[va_g][group_col].tolist())

        tr_idx = df_wide[df_wide[group_col].isin(tr_groups)].index.values
        va_idx = df_wide[df_wide[group_col].isin(va_groups)].index.values
        splits.append((tr_idx, va_idx))

    return splits


# =========================================================
# 6) Dataset / Augment
# =========================================================
class BiomassDataset(Dataset):
    def __init__(
        self,
        df_wide: pd.DataFrame,
        img_dir: str,
        target_cols: List[str],
        transform,
        mode: str,
    ):
        self.df = df_wide.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.target_cols = target_cols
        self.transform = transform
        self.mode = mode  # "train" / "valid"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        full_path = self.img_dir / img_path

        image = cv2.imread(str(full_path))
        if image is None:
            raise FileNotFoundError(f"画像が読めません: {full_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        out = {"image": image, "image_path": img_path}

        targets = row[self.target_cols].values.astype(np.float32)
        out["targets"] = torch.tensor(targets, dtype=torch.float32)
        return out


def build_transforms(aug_cfg: dict, mode: str, default_size: int = 224):
    """
    aug_cfg は例として以下の形式を想定:
      augmentation:
        train:
          Resize: {height: 224, width: 224}
          HorizontalFlip: {p: 0.5}
          Normalize: {mean: [...], std: [...]}
        valid:
          Resize: {height: 224, width: 224}
          Normalize: {mean: [...], std: [...]}
    """
    mode_cfg = (aug_cfg or {}).get(mode, {}) or {}

    tfs: List[A.BasicTransform] = []
    has_resize = False

    # よく使うものだけ対応（不足があればここに足す）
    for name, params in mode_cfg.items():
        if name == "Resize":
            tfs.append(A.Resize(**params))
            has_resize = True
        elif name == "RandomResizedCrop":
            tfs.append(A.RandomResizedCrop(**params))
            has_resize = True
        elif name == "HorizontalFlip":
            tfs.append(A.HorizontalFlip(**params))
        elif name == "VerticalFlip":
            tfs.append(A.VerticalFlip(**params))
        elif name == "ShiftScaleRotate":
            tfs.append(A.ShiftScaleRotate(**params))
        elif name == "RandomBrightnessContrast":
            tfs.append(A.RandomBrightnessContrast(**params))
        elif name == "ColorJitter":
            tfs.append(A.ColorJitter(**params))
        elif name == "HueSaturationValue":
            tfs.append(A.HueSaturationValue(**params))
        elif name == "CoarseDropout":
            tfs.append(A.CoarseDropout(**params))
        elif name == "Normalize":
            tfs.append(A.Normalize(**params))

    if not has_resize:
        tfs.insert(0, A.Resize(default_size, default_size))

    tfs.append(ToTensorV2())
    return A.Compose(tfs)


def create_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    img_dir: str,
    target_cols: List[str],
    aug_cfg: dict,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_tf = build_transforms(aug_cfg, mode="train")
    valid_tf = build_transforms(aug_cfg, mode="valid")

    train_ds = BiomassDataset(train_df, img_dir=img_dir, target_cols=target_cols, transform=train_tf, mode="train")
    valid_ds = BiomassDataset(valid_df, img_dir=img_dir, target_cols=target_cols, transform=valid_tf, mode="valid")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, valid_loader


# =========================================================
# 7) Model（Lightning）
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
        total_steps: Optional[int],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_ratio = float(warmup_ratio)
        self.total_steps = total_steps

        # loss weight（ターゲット順に合わせる）
        self.loss_weights = torch.tensor(list(loss_weights.values()), dtype=torch.float32)

        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            # 保険（ほぼ通らない想定）
            feat_dim = 768

        self.head = nn.Sequential(
            nn.Linear(int(feat_dim), 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_targets),
        )

        self.criterion = nn.MSELoss(reduction="none")

        self.ema = None
        self.ema_enabled = bool(ema_enabled) and _HAS_EMA
        self.ema_decay = float(ema_decay)

        self._val_cache = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["targets"]
        pred = self(x)

        loss_mat = self.criterion(pred, y)  # (B, T)
        w = self.loss_weights.to(loss_mat.device)
        loss = (loss_mat * w).sum() / w.sum()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["targets"]
        pred = self(x)

        loss_mat = self.criterion(pred, y)
        w = self.loss_weights.to(loss_mat.device)
        loss = (loss_mat * w).sum() / w.sum()

        self._val_cache.append(
            {
                "pred": pred.detach().cpu(),
                "y": y.detach().cpu(),
                "image_path": batch["image_path"],
            }
        )
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if len(self._val_cache) == 0:
            return
        pred = torch.cat([d["pred"] for d in self._val_cache], dim=0)
        y = torch.cat([d["y"] for d in self._val_cache], dim=0)

        mae = (pred - y).abs().mean(dim=0)
        for i, v in enumerate(mae.tolist()):
            self.log(f"val_mae_t{i}", v, prog_bar=False)

        self._val_cache.clear()

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.ema_enabled:
            self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)

        if self.total_steps is None:
            return opt

        warmup_steps = int(self.total_steps * self.warmup_ratio)
        sch = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=warmup_steps,
            num_training_steps=int(self.total_steps),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update()

    def on_train_end(self):
        if self.ema is not None:
            self.ema.copy_to()


# =========================================================
# 8) Train fold
# =========================================================
def train_one_fold(
    fold: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    cfg: dict,
    exp_cfg: dict,
    artifacts_dir: Path,
    logger,
) -> pd.DataFrame:
    fold_dir = artifacts_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # dataloader
    train_loader, valid_loader = create_dataloaders(
        train_df=train_df,
        valid_df=valid_df,
        img_dir=str(Path(cfg["data"]["input_dir"])),
        target_cols=cfg["targets"]["primary"],
        aug_cfg=exp_cfg.get("augmentation", {}),
        batch_size=int(exp_cfg["training"]["batch_size"]),
        num_workers=int(exp_cfg["training"]["num_workers"]),
    )

    total_steps = len(train_loader) * int(exp_cfg["training"]["epochs"])

    model = BiomassModel(
        backbone=exp_cfg["model"]["backbone"],
        num_targets=int(exp_cfg["model"]["num_targets"]),
        lr=float(exp_cfg["optimizer"]["lr"]),
        weight_decay=float(exp_cfg["optimizer"]["weight_decay"]),
        warmup_ratio=float(exp_cfg["scheduler"]["warmup_ratio"]),
        loss_weights=exp_cfg["training"]["loss_weights"],
        ema_enabled=bool(exp_cfg["ema"]["enabled"]),
        ema_decay=float(exp_cfg["ema"]["decay"]),
        total_steps=total_steps,
    )

    # logger（W&B が無い環境でも落ちないようにする）
    pl_logger = None
    if _HAS_WANDB_LOGGER and ("logging" in cfg) and ("wandb" in cfg["logging"]):
        wcfg = cfg["logging"]["wandb"]
        try:
            pl_logger = WandbLogger(
                name=f"{cfg['output']['exp_name']}_{exp_cfg['exp_id']}_fold{fold}",
                project=wcfg.get("project"),
                entity=wcfg.get("entity"),
                save_dir=str(artifacts_dir / "wandb"),
                offline=(wcfg.get("mode") == "offline"),
            )
        except Exception as e:
            logger.warning(f"WandbLogger を初期化できませんでした: {e}. ロガー無しで続行します。")
            pl_logger = None

    ckpt_cfg = exp_cfg["callbacks"]["model_checkpoint"]
    callbacks = [
        ModelCheckpoint(
            dirpath=str(fold_dir),
            filename=str(ckpt_cfg["filename"]),
            monitor=str(ckpt_cfg["monitor"]),
            mode=str(ckpt_cfg["mode"]),
            save_top_k=int(ckpt_cfg["save_top_k"]),
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=int(exp_cfg["training"]["epochs"]),
        logger=pl_logger,
        callbacks=callbacks,
        precision=16 if bool(exp_cfg.get("mixed_precision", False)) else 32,
        gradient_clip_val=float(exp_cfg.get("gradient_clip_val", 0.0)),
        deterministic=True,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
    )

    with trace(f"Training fold {fold}"):
        trainer.fit(model, train_loader, valid_loader)

    # OOF（primary だけ集める）
    model.eval()
    device = model.device
    preds_all = []
    trues_all = []
    paths_all: List[str] = []

    with torch.no_grad():
        for batch in valid_loader:
            x = batch["image"].to(device)
            y = batch["targets"].cpu().numpy()
            p = model(x).detach().cpu().numpy()
            preds_all.append(p)
            trues_all.append(y)
            paths_all.extend(batch["image_path"])

    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)

    # wide 形式 OOF（primary の true/pred）
    prim = cfg["targets"]["primary"]
    oof = pd.DataFrame({"image_path": paths_all})
    for i, t in enumerate(prim):
        oof[f"{t}_true"] = trues_all[:, i]
        oof[f"{t}_pred"] = preds_all[:, i]
    oof["fold"] = fold

    return oof


# =========================================================
# 9) main
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

    logger = get_logger(file_name=f"train_{exp_name}_{exp_id}", file_dir=log_dir)
    logger.info(f"Artifacts dir: {artifacts_dir}")

    # load train data
    train_csv = Path(cfg["data"]["input_dir"]) / str(cfg["data"]["train_csv"])
    with trace("Loading train.csv"):
        raw_df = pd.read_csv(train_csv)
        logger.info(f"Loaded train rows: {len(raw_df)} from {train_csv}")

    group_col = str(cfg["cv"]["group_col"])
    stratify_col = str(cfg["cv"]["stratify_col"])
    n_folds = int(cfg["cv"]["n_folds"])
    cv_seed = int(cfg["cv"]["seed"])

    with trace("Normalize train dataframe to wide"):
        df_wide = to_wide_train_df(
            raw_df=raw_df,
            target_primary=cfg["targets"]["primary"],
            group_col=group_col,
            stratify_col=stratify_col,
        )
        logger.info(f"Wide train rows (unique images): {len(df_wide)}")

    with trace("Creating CV splits"):
        splits = stratified_group_kfold(
            df_wide=df_wide,
            n_folds=n_folds,
            group_col=group_col,
            stratify_col=stratify_col,
            seed=cv_seed,
        )
        logger.info(f"Splits: {len(splits)} folds")

    # train folds
    oof_list: List[pd.DataFrame] = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        logger.info(f"Fold {fold}: train={len(tr_idx)} valid={len(va_idx)}")
        tr_df = df_wide.iloc[tr_idx].reset_index(drop=True)
        va_df = df_wide.iloc[va_idx].reset_index(drop=True)

        oof_fold = train_one_fold(
            fold=fold,
            train_df=tr_df,
            valid_df=va_df,
            cfg=cfg,
            exp_cfg=exp_cfg,
            artifacts_dir=artifacts_dir,
            logger=logger,
        )
        oof_list.append(oof_fold)

    with trace("Save OOF and compute metrics"):
        oof_wide = pd.concat(oof_list, ignore_index=True)

        # true/pred の secondary を作る（wide で作ってから long 化）
        # true 側
        true_df = pd.DataFrame({
            "image_path": oof_wide["image_path"].values,
            "Dry_Total_g": oof_wide["Dry_Total_g_true"].values,
            "GDM_g": oof_wide["GDM_g_true"].values,
            "Dry_Green_g": oof_wide["Dry_Green_g_true"].values,
        })
        true_df = derive_secondary_from_primary(true_df)
        # pred 側
        pred_df = pd.DataFrame({
            "image_path": oof_wide["image_path"].values,
            "Dry_Total_g": oof_wide["Dry_Total_g_pred"].values,
            "GDM_g": oof_wide["GDM_g_pred"].values,
            "Dry_Green_g": oof_wide["Dry_Green_g_pred"].values,
        })
        pred_df = derive_secondary_from_primary(pred_df)

        # long（metrics用）
        all_targets = cfg["targets"]["all"]
        rows = []
        for i in range(len(oof_wide)):
            img = oof_wide.loc[i, "image_path"]
            img_id = Path(str(img)).stem
            for t in all_targets:
                rows.append({
                    "sample_id": f"{img_id}__{t}",
                    "image_path": img,
                    "fold": int(oof_wide.loc[i, "fold"]),
                    "target_name": t,
                    "target": float(true_df.loc[i, t]),
                    "pred": float(pred_df.loc[i, t]),
                })
        oof_long = pd.DataFrame(rows)

        # 保存
        oof_dir = artifacts_dir / "OOF"
        oof_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        oof_path = oof_dir / f"OOF_{exp_name}_{exp_id}_{ts}.csv"
        oof_long.to_csv(oof_path, index=False)
        logger.info(f"Saved OOF (long) to: {oof_path}")

        # metrics
        metrics = calculate_oof_metrics(oof_long)
        logger.info("=" * 60)
        logger.info("OOF metrics")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.6f}")
        logger.info("=" * 60)

        # meta.yaml
        meta = {
            "exp_name": exp_name,
            "exp_id": exp_id,
            "timestamp": ts,
            "metrics": metrics,
        }
        meta_path = artifacts_dir / "meta.yaml"
        with open(meta_path, "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)
        logger.info(f"Saved meta to: {meta_path}")


if __name__ == "__main__":
    main()
