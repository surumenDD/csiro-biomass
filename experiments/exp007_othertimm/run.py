from utils.timing import trace
from utils.logger import get_logger
from utils.env import EnvConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
import hydra
import wandb  # main関数のwandb.init用
import timm
import os
import sys

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold, StratifiedKFold
from PIL import Image
from tqdm.auto import tqdm
tqdm.pandas()


LOGGER = None

# ----Utils----


# ----Config----
@dataclass
class ExpConfig:
    debug: bool = False
    accelerator: str = "auto"
    devices: str = "auto"
    log_every_n_steps: int = 5

    # meta
    exp_name: str = "exp002_effnet-b2"

    # model
    model_name: str = "efficientnet_b2"
    image_size: int = 260
    n_folds: int = 5

    # training
    seed: int = 42
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 10
    weight_decay: float = 0.0
    num_workers: int = 4
    # scheduler_name: str = "cosine"
    scheduler_name: str = "plateau"
    t_max: int = 400
    min_lr: float = 1e-6

    plateau_factor: float = 0.95
    plateau_patience: int = 8
    plateau_cooldown: int = 0



    # wandb
    wandb_project: str = "csiro-biomass"
    wandb_entity: Optional[str] = None  # 個人ならNone
    wandb_mode: str = "online"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)


# hydra用にdefaultを設定
# YAMLで両者を合成する
cs = ConfigStore.instance()
cs.store(name="default", group="env", node=EnvConfig)
cs.store(name="default", group="exp", node=ExpConfig)


# ----実験用コード----
def log_config(cfg: Config) -> None:
    LOGGER.info("Config: %s", cfg)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_input_dir(cfg_input_dir: str) -> Path:
    # Primary: from config
    cfg_dir = Path(cfg_input_dir)
    if cfg_dir.exists():
        return cfg_dir
    # Secondary: local workspace input
    workspace_dir = Path(__file__).resolve().parents[2] / "input"
    if workspace_dir.exists():
        return workspace_dir
    # Fallback (e.g. Kaggle): current working directory input
    cwd_input_dir = Path.cwd() / "input"
    return cwd_input_dir


class ImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        input_dir: Path,
        transform: Optional[transforms.Compose] = None,
        has_targets: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.input_dir = input_dir
        self.transform = transform
        self.has_targets = has_targets

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.input_dir / row["image_path"]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if not self.has_targets:
            return image
        targets = torch.tensor(
            [row["Dry_Green_g"], row["Dry_Clover_g"], row["Dry_Dead_g"]],
            dtype=torch.float32,
        )
        return image, targets


def create_transforms(image_size: int, aug: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
    if not aug:
        return transforms.Compose(base), transforms.Compose(base)

    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    )
    valid_tfms = transforms.Compose(base)
    return train_tfms, valid_tfms


# 学習で直接予測する3つ
PRED3_COLS = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g"]

# コンペ評価の5つ
TARGET5_ORDER = ["Dry_Green_g", "Dry_Clover_g",
                 "Dry_Dead_g", "GDM_g", "Dry_Total_g"]

META_COLS = [
    "sample_id_prefix",
    "image_path",
    "Sampling_Date",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
]


def make_train_wide(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    df[["sample_id_prefix", "sample_id_suffix"]
       ] = df["sample_id"].str.split("__", expand=True)

    wide = (
        df.pivot_table(
            index=META_COLS,
            columns="target_name",
            values="target",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    return wide


def make_test_tables(test_csv: Path):
    test_long = pd.read_csv(test_csv)
    test_long[["sample_id_prefix", "sample_id_suffix"]
              ] = test_long["sample_id"].str.split("__", expand=True)

    # 画像1枚=1行（推論用）
    test_img = (
        test_long[["image_path", "sample_id_prefix"]]
        .drop_duplicates(subset=["image_path"])
        .reset_index(drop=True)
    )
    return test_img, test_long


def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    y_true, y_pred: (N, 5) = Green/Clover/Dead/GDM/Total
    """
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)
    r2_scores = []  # 各ターゲットのR^2を入れるリスト
    for i in range(5):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    weighted_r2 = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted_r2, r2_scores


def calc_metric(outputs3: np.ndarray, targets3: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    3ターゲットを5ターゲットに復元（コンペの評価）
    """
    y_true = np.column_stack([
        targets3,
        targets3[:, :2].sum(axis=1),  # GDM_g
        targets3.sum(axis=1),  # Dry_Total_g
    ])
    y_pred = np.column_stack([
        outputs3,
        outputs3[:, :2].sum(axis=1),
        outputs3.sum(axis=1),
    ])
    return weighted_r2_score(y_true, y_pred)


def create_model(model_name: str, num_classes: int = 3) -> nn.Module:
    # timmモデルを使用
    try:
        model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes, in_chans=3)
    except Exception as e:
        raise ValueError(f"Failed to create timm model '{model_name}':{e}")
    return model


class RegressionModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float,
        weight_decay: float,
        scheduler_name: str = "cosine",
        t_max: int = 100,
        min_lr: float = 1e-6,
        plateau_factor: float = 0.5,
        plateau_patience: int = 2,
        plateau_cooldown: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model_name=model_name, num_classes=3)
        self.loss_fn = nn.SmoothL1Loss()
        self._val_preds = []
        self._val_targets = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, targets = batch
        preds = self(images)
        loss = self.loss_fn(preds, targets)
        targets_f = targets.to(torch.float32)
        preds_f = preds.to(torch.float32)
        # 参考値としてrmse
        preds_clamped = torch.clamp(preds_f, min=0.0)
        rmse = torch.sqrt(torch.mean((preds_clamped - targets_f) ** 2))
        # log
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("train_rmse", rmse, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        images, targets = batch
        preds = self(images)
        loss = self.loss_fn(preds, targets)
        targets_f = targets.to(torch.float32)
        preds_f = preds.to(torch.float32)

        preds_clamped = torch.clamp(preds_f, min=0.0)
        rmse = torch.sqrt(torch.mean((preds_clamped - targets_f) ** 2))

        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)

        # metricようにCPUに移して溜める(epoch endでまとめて計算)
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets_f.detach().cpu().numpy()

        self._val_preds.append(preds_np)
        self._val_targets.append(targets_np)

    def on_validation_epoch_end(self) -> None:
        if len(self._val_preds) == 0:
            return

        preds3 = np.concatenate(self._val_preds, axis=0)     # (N, 3)
        targs3 = np.concatenate(self._val_targets, axis=0)   # (N, 3)

        weighted_r2, r2_each = calc_metric(preds3, targs3)

        self.log("val_weighted_r2", float(weighted_r2),
                 on_epoch=True, prog_bar=True)
        self.log("val_r2_green", float(r2_each[0]), prog_bar=False)
        self.log("val_r2_clover", float(r2_each[1]), prog_bar=False)
        self.log("val_r2_dead",   float(r2_each[2]), prog_bar=False)
        self.log("val_r2_gdm",   float(r2_each[3]), prog_bar=False)
        self.log("val_r2_total", float(r2_each[4]), prog_bar=False)

        self._val_preds.clear()
        self._val_targets.clear()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: Optional[int] = 0):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            images, _ = batch
        else:
            images = batch

        preds = self(images).to(torch.float32)
        # preds = torch.clamp(preds, min=0.0)
        return preds

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        name = getattr(self.hparams, "scheduler_name", "cosine")

        # --- 追加：metric依存（おすすめ） ---
        if name in ("plateau", "reduce_on_plateau", "rlrop"):
            # cfgに項目が無い場合でも動くように、getattrで安全に読む
            factor = float(getattr(self.hparams, "plateau_factor", 0.5))
            patience = int(getattr(self.hparams, "plateau_patience", 2))
            min_lr = float(getattr(self.hparams, "min_lr", 1e-6))
            cooldown = int(getattr(self.hparams, "plateau_cooldown", 0))

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",                 # val_weighted_r2 は大きいほど良い
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                cooldown=cooldown,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_weighted_r2",  # ここが肝
                    "frequency": 1,
                    "strict": True,
                },
            }

        # --- 既存：cosine ---
        if name in ("cosine", "coslr", "cosine_annealing"):
            t_max = int(getattr(self.hparams, "t_max", 100))
            eta_min = float(getattr(self.hparams, "min_lr", 0.0))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=eta_min)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        return optimizer


def train_one_fold(
    fold: int,
    train_df: pd.DataFrame,
    input_path: Path,
    cfg: Config,
    output_dir: Path,
    exp_name: str,
):
    pl.seed_everything(cfg.exp.seed + fold, workers=True)
    train_tfms, valid_tfms = create_transforms(cfg.exp.image_size, aug=True)
    epochs = 1 if cfg.exp.debug else cfg.exp.epochs

    trn_idx = np.where(train_df["fold"].values != fold)[0]
    val_idx = np.where(train_df["fold"].values == fold)[0]

    trn_ds = ImageDataset(
        df=train_df.iloc[trn_idx],
        input_dir=input_path,
        transform=train_tfms,
        has_targets=True,
    )
    val_ds = ImageDataset(
        df=train_df.iloc[val_idx],
        input_dir=input_path,
        transform=valid_tfms,
        has_targets=True,
    )

    trn_loader = DataLoader(
        trn_ds,
        batch_size=cfg.exp.batch_size,
        shuffle=True,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.exp.batch_size,
        shuffle=False,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    module = RegressionModule(
        model_name=cfg.exp.model_name,
        learning_rate=cfg.exp.learning_rate,
        weight_decay=cfg.exp.weight_decay,
        scheduler_name=cfg.exp.scheduler_name,
        t_max=cfg.exp.t_max,
        min_lr=cfg.exp.min_lr,
        plateau_factor=cfg.exp.plateau_factor,
        plateau_patience=cfg.exp.plateau_patience,
        plateau_cooldown=cfg.exp.plateau_cooldown,
    )

    accelerator = cfg.exp.accelerator
    devices = cfg.exp.devices

    # ckpt_cb = ModelCheckpoint(
    #     dirpath=str(output_dir),
    #     filename=f"model_fold{fold}-best",
    #     monitor="val_weighted_r2",
    #     mode="max",
    #     save_top_k=1,
    #     save_weights_only=False,

    # )
    ckpt_best = ModelCheckpoint(
        dirpath=str(output_dir),
        filename=f"model_fold{fold}-best",
        monitor="val_weighted_r2",
        mode="max",
        save_top_k=1,
        save_weights_only=False,
    )

    ckpt_last = ModelCheckpoint(
        dirpath=str(output_dir),
        filename=f"model_fold{fold}-last",
        save_last=True,
        save_top_k=1,
    )

    lr_cb = LearningRateMonitor(logging_interval="epoch")

    if cfg.exp.debug:
        os.environ["WANDB_MODE"] = "disabled"
    # WandBの設定を変更
    wandb.finish()
    wandb_logger = WandbLogger(
        project=cfg.exp.wandb_project,
        group=exp_name,               # 実験名ごとにグループ化
        name=f"{exp_name}_fold-{fold}",  # 表示名を「実験名_fold-0」にする
        job_type="train",
        # 各Runに設定情報を紐付ける
        config=OmegaConf.to_container(cfg.exp, resolve=True),
        reinit=True
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=[ckpt_best, ckpt_last, lr_cb],
        enable_checkpointing=True,
        deterministic=True,
        log_every_n_steps=cfg.exp.log_every_n_steps,
    )

    trainer.fit(module, train_dataloaders=trn_loader, val_dataloaders=val_loader)

    best_path = ckpt_best.best_model_path
    last_path = ckpt_last.last_model_path
    LOGGER.info(f"Fold {fold} best checkpoint: {best_path}")
    LOGGER.info(f"Fold {fold} last checkpoint: {last_path}")

    # ===== OOF（3列）を作る =====
    best_module = RegressionModule.load_from_checkpoint(best_path)
    pred_batches = trainer.predict(best_module, dataloaders=val_loader)

    oof_pred = torch.cat(pred_batches, dim=0).detach(
    ).cpu().numpy().astype(np.float32)  # (N,3)

    # 真値（3列）も同じ順で取る（ここが PRED3_COLS の順番保証）
    oof_true = train_df.iloc[val_idx][PRED3_COLS].values.astype(
        np.float32)              # (N,3)

    # ===== CV指標（weighted R2） =====
    fold_weighted_r2, fold_r2_each = calc_metric(oof_pred, oof_true)

    # 参考値：3列のRMSE
    fold_rmse = float(
        np.sqrt(np.mean((np.maximum(oof_pred, 0.0) - oof_true) ** 2)))

    LOGGER.info(f"Fold {fold} OOF weighted_r2: {fold_weighted_r2:.6f}")
    LOGGER.info(f"Fold {fold} OOF rmse(3): {fold_rmse:.6f}")

    wandb.finish()

    return oof_pred, fold_weighted_r2, {"best": best_path, "last": last_path}


def make_folds(
    train_wide: pd.DataFrame,
    n_splits: int,
    seed: int,
    strat_col: str = "State",
    group_col: str = "Sampling_Date",
) -> pd.DataFrame:
    df = train_wide.copy()
    df["fold"] = -1

    # sgkf = StratifiedGroupKFold(
    #     n_splits=n_splits, shuffle=True, random_state=seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed) 

    X = np.zeros(len(df))  # 使わないがAPI上必要
    y = df[strat_col].astype(str).values
    groups = df[group_col].astype(str).values

    for fold, (_, val_idx) in enumerate(skf.split(X=X, y=y)):
        df.loc[val_idx, "fold"] = fold

    return df


def _score_split_balance(
    df_with_fold: pd.DataFrame,
    strat_col: str = "State",
    pred3_cols: list[str] = None,
) -> float:
    # 小さいほど良い（0が理想）
    n = len(df_with_fold)
    fold_counts = df_with_fold["fold"].value_counts().sort_index()
    size_ratio = fold_counts.max() / max(1, fold_counts.min())  # 1が理想

    # State分布のズレ（L1距離の平均）
    overall = df_with_fold[strat_col].value_counts(normalize=True)
    l1_sum = 0.0
    for f in sorted(df_with_fold["fold"].unique()):
        dist = df_with_fold.loc[df_with_fold["fold"] ==
                                f, strat_col].value_counts(normalize=True)
        dist = dist.reindex(overall.index, fill_value=0.0)
        l1_sum += float(np.abs(dist.values - overall.values).sum())
    state_l1 = l1_sum / max(1, df_with_fold["fold"].nunique())

    # ターゲット平均のズレ（任意）
    target_pen = 0.0
    if pred3_cols is not None and all(c in df_with_fold.columns for c in pred3_cols):
        global_mean = df_with_fold[pred3_cols].mean()
        per = df_with_fold.groupby("fold")[pred3_cols].mean()
        target_pen = float(np.abs(per - global_mean).mean().mean())  # 小さいほど良い

    # 合成（重みは好みで調整）
    score = (size_ratio - 1.0) * 1.0 + state_l1 * 2.0 + target_pen * 1.0
    return score


def make_folds_with_seed_search(
    train_wide: pd.DataFrame,
    n_splits: int,
    seed_candidates: list[int],
    strat_col: str = "State",
    group_col: str = "Sampling_Date",
    pred3_cols: list[str] = None,
) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    best_seed = None
    best_df = None
    best_score = None
    rows = []

    for seed in seed_candidates:
        df = make_folds(
            train_wide=train_wide,
            n_splits=n_splits,
            seed=seed,
            strat_col=strat_col,
            group_col=group_col,
        )
        score = _score_split_balance(
            df, strat_col=strat_col, pred3_cols=pred3_cols)
        rows.append({"seed": seed, "score": score})

        if best_score is None or score < best_score:
            best_score = score
            best_seed = seed
            best_df = df

    score_df = pd.DataFrame(rows).sort_values(
        "score", ascending=True).reset_index(drop=True)
    return best_df, best_seed, score_df


@torch.no_grad()
def predict_test(
    fold_to_ckpt: dict[int, str],   # fold -> ckpt path
    input_dir: Path,
    test_csv: Path,
    sample_sub_csv: Path,
    cfg: Config,
) -> pd.DataFrame:
    # 1) test を (画像1枚=1行) と (sample_id単位のlong) に分ける
    test_img, test_long = make_test_tables(test_csv)

    # 2) 推論用 transform / dataset / loader
    _, valid_tfms = create_transforms(cfg.exp.image_size, aug=False)
    test_ds = ImageDataset(
        df=test_img,
        input_dir=input_dir,
        transform=valid_tfms,
        has_targets=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.exp.batch_size,
        shuffle=False,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 3) fold 平均で pred3 (N,3) を作る
    pred_sum = None
    n_folds = 0

    # 重要：foldの順は固定（再現性・デバッグ性のため）
    for fold in sorted(fold_to_ckpt.keys()):
        ckpt_path = fold_to_ckpt[fold]

        module = RegressionModule.load_from_checkpoint(ckpt_path)
        module.eval()

        trainer = pl.Trainer(
            accelerator=cfg.exp.accelerator,
            devices=cfg.exp.devices,   # "auto"
            logger=False,
            enable_checkpointing=False,
        )

        pred_batches = trainer.predict(module, dataloaders=test_loader)
        pred3 = torch.cat(pred_batches, dim=0).detach(
        ).cpu().numpy().astype(np.float32)  # (N,3)

        pred_sum = pred3 if pred_sum is None else (pred_sum + pred3)
        n_folds += 1

    pred3 = pred_sum / max(1, n_folds)  # (N,3) = [Green, Clover, Dead]

    # 4) 3 -> 5 復元（A案：足し算）
    pred_green = pred3[:, 0]
    pred_clover = pred3[:, 1]
    pred_dead = pred3[:, 2]
    pred_gdm = pred_green + pred_clover
    pred_total = pred_green + pred_clover + pred_dead

    pred_wide = test_img.copy()
    pred_wide["Dry_Green_g"] = pred_green
    pred_wide["Dry_Clover_g"] = pred_clover
    pred_wide["Dry_Dead_g"] = pred_dead
    pred_wide["GDM_g"] = pred_gdm
    pred_wide["Dry_Total_g"] = pred_total

    # 5) sample_submission の行順を基準に target を埋める
    sub = pd.read_csv(sample_sub_csv)  # columns: [sample_id, target]

    # sample_id -> (prefix/suffix) を作る
    sub[["sample_id_prefix", "sample_id_suffix"]
        ] = sub["sample_id"].str.split("__", expand=True)

    # prefix -> image_path を付与（test_long を使う）
    # sample_id_prefix は同一prefixで同一image_pathの前提
    prefix_to_img = (
        test_long[["sample_id_prefix", "image_path"]]
        .drop_duplicates(subset=["sample_id_prefix"])
        .reset_index(drop=True)
    )
    sub = sub.merge(prefix_to_img, on="sample_id_prefix", how="left")

    # image_path -> 予測5列を付与
    sub = sub.merge(
        pred_wide[["image_path"] + TARGET5_ORDER],
        on="image_path",
        how="left",
    )

    # suffix に応じて target を決める（行順は sub のまま）
    sub["target"] = 0.0
    for col in TARGET5_ORDER:
        m = sub["sample_id_suffix"] == col
        sub.loc[m, "target"] = sub.loc[m, col].astype(float)

    return sub[["sample_id", "target"]]


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:
    print(cfg)

    exp_name = f"{Path(sys.argv[0]).parent.name}/{HydraConfig.get().runtime.choices.exp}"
    output_dir = Path(cfg.env.artifacts_dir) / exp_name
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    global LOGGER
    LOGGER = get_logger(__name__, output_dir)
    LOGGER.info("Start")
    log_config(cfg)

    # resolve paths
    input_dir = resolve_input_dir(cfg.env.input_dir)
    # hoge
    train_csv = input_dir / "train.csv"
    test_csv = input_dir / "test.csv"
    sample_submission_csv = input_dir / "sample_submission.csv"

    # seed
    set_seed(cfg.exp.seed)

    # 5070ti用 #kaggle環境ではコメントアウト
    torch.set_float32_matmul_precision('high')

    # load data
    with trace("load.csv"):
        train_wide = make_train_wide(train_csv)
        train_wide["image_path"] = train_wide["image_path"].astype(str)
        train_wide["sample_id_prefix"] = train_wide["sample_id_prefix"].astype(
            str)
        
    with trace("make_folds"):
        train_wide = make_folds(
            train_wide=train_wide,
            n_splits=cfg.exp.n_folds,
            seed=cfg.exp.seed,
            strat_col="State",
            group_col="Sampling_Date",
        )

    # with trace("make_folds"):
    #     seed_candidates = list(range(0, 200))  # 例：0〜199 を探索（好みで増減）
    #     train_wide, best_seed, score_df = make_folds_with_seed_search(
    #         train_wide=train_wide,
    #         n_splits=cfg.exp.n_folds,
    #         seed_candidates=seed_candidates,
    #         strat_col="State",
    #         group_col="Sampling_Date",
    #         pred3_cols=PRED3_COLS,
    #     )
    #     LOGGER.info(f"Best fold seed: {best_seed}")
    #     LOGGER.info("fold counts: %s", train_wide["fold"].value_counts().to_dict())
    #     score_df.to_csv(output_dir / "fold_seed_search.csv", index=False)

    # return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    # ===== train per fold =====
    n = len(train_wide)
    oof_pred_full = np.zeros((n, 3), dtype=np.float32)  # (N,3) = PRED3_COLS順
    fold_scores: dict[int, float] = {}
    fold_to_ckpt: dict[int, str] = {}

    folds_all = sorted(train_wide["fold"].unique().tolist())
    folds_run = getattr(cfg.exp, "folds", folds_all)
    # 安全策：範囲外があれば全foldに戻す（あなたの意図どおり事故防止）
    if any(f not in folds_all for f in folds_run):
        LOGGER.warning(
            "Some requested folds are out of range; defaulting to all folds")
        folds_run = folds_all

    for fold in folds_run:
        with trace(f"train_fold_{fold}"):
            oof_pred_fold, fold_weighted_r2, ckpt = train_one_fold(
                fold=fold,
                train_df=train_wide,
                input_path=input_dir,
                cfg=cfg,
                output_dir=output_dir,
                exp_name=exp_name,
            )

        val_mask = (train_wide["fold"].values == fold)
        if oof_pred_fold.shape[0] != int(val_mask.sum()):
            raise RuntimeError(
                f"OOF size mismatch: fold={fold}, "
                f"oof_pred_fold={oof_pred_fold.shape[0]}, val_rows={int(val_mask.sum())}"
            )

        oof_pred_full[val_mask] = oof_pred_fold.astype(np.float32)
        fold_scores[int(fold)] = float(fold_weighted_r2)
        fold_to_ckpt[int(fold)] = str(ckpt["best"])

        LOGGER.info(f"Fold {fold} weighted_r2: {fold_weighted_r2:.6f}")
        LOGGER.info(f"Fold {fold} best checkpoint: {ckpt['best']}")
        LOGGER.info(f"Fold {fold} last checkpoint: {ckpt['last']}")

    # OOF scoring
    oof_true_full = train_wide[PRED3_COLS].values.astype(np.float32)  # (N,3)
    if train_wide[PRED3_COLS].isna().any().any():
        raise ValueError(
            "NaN found in PRED3_COLS after pivot. Check train.csv completeness.")

    oof_weighted_r2, oof_r2_each = calc_metric(oof_pred_full, oof_true_full)

    LOGGER.info(f"OOF weighted_r2: {oof_weighted_r2:.6f}")
    LOGGER.info(
        "OOF r2_each: green=%.6f clover=%.6f dead=%.6f gdm=%.6f total=%.6f",
        float(oof_r2_each[0]),
        float(oof_r2_each[1]),
        float(oof_r2_each[2]),
        float(oof_r2_each[3]),
        float(oof_r2_each[4]),
    )
    # Save OOF
    oof_df = train_wide[["sample_id_prefix", "image_path"] + PRED3_COLS].copy()
    oof_df[["pred_green", "pred_clover", "pred_dead"]] = oof_pred_full

    oof_path = output_dir / "oof.csv"
    oof_df.to_csv(oof_path, index=False)
    LOGGER.info(f"Saved OOF to {oof_path}")

    # Predict test
    with trace("predict_test"):
        submission = predict_test(
            fold_to_ckpt=fold_to_ckpt,
            input_dir=input_dir,
            test_csv=test_csv,
            sample_sub_csv=sample_submission_csv,
            cfg=cfg,
        )
        sub_path = output_dir / "submission.csv"
        submission.to_csv(sub_path, index=False)
        LOGGER.info(f"Saved submission to {sub_path}")


if __name__ == "__main__":
    main()
