import os
import sys

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from PIL import Image
from tqdm.auto import tqdm
tqdm.pandas()

import timm

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

from utils.env import EnvConfig
from utils.logger import get_logger

LOGGER = None

# ----Utils----


# ----Config----
@dataclass
class ExpConfig:
    debug: bool = True
    accelerator: str = "auto"
    devices: str = "auto"
    log_every_n_steps: int = 20

    # meta
    exp_name: str = "exp002_effnet-b2"

    # model
    model_name: str = "efficientnet_b2"
    image_size: int = 260
    n_folds: int = 5

    # training
    seed: int = 42
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 10
    weight_decay: float = 0.0
    num_workers: int = 4

    # wandb
    wandb_project: str = "csiro-biomass"
    wandb_entity: Optional[str] = None # 個人ならNone
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
    torch.backends.cudnn.deterministic =True
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


class RegressionDataset(Dataset):
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
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        if not self.has_targets:
            return image

        targets = torch.tensor(
            [row[Dry_Green_g], row["Dry_Clover_g"], row["Dry_Dead_g"]],
            dtype=torch.float32,
        )
        return image, targets


def create_transforms(image_size: int, aug: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    valid_tfms = transforms.Compose(base)
    return train_tfms, valid_tfms
    


def make_wide_train_df(train_csv: Path) -> pd.DataFrame:
    """
    train.csv は (sample_id, target_name, target, image_path, ...) のロング形式。
    画像1枚に対して 5ターゲットが縦持ちなので、画像単位で横持ち(wide)に変換する。
    """
    train_df = pd.read_csv(train_csv)
    train_df[["sample_id_prefix", "sample_id_suffix"]] = train_df["sample_id"].str.split("__", expand=True)

    cols = [
        "sample_id_prefix",
        "image_path",
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
    ]
    wide = train_df.groupby(cols).apply(lambda d: d.set_index("target_name")["target"])
    wide = wide.reset_index()
    wide.columns.name = None

    # 念のため型
    for c in ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]:
        if c in wide.columns:
            wide[c] = wide[c].astype(np.float32)
    return wide

###ここからしたコミットしていない

def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    y_true, y_pred: (N, 5) = Green/Clover/Dead/GDM/Total
    """
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)
    r2_scores = [] # 各ターゲットのR^2を入れるリスト
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

    # seed
    set_seed(cfg.exp.seed)

    # paths
    input_dir = resolve_input_dir(cfg.env.input_dir)

if __name__ == "__main__":
    main()