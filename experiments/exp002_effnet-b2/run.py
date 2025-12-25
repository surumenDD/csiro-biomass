import os

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


# Utils


# Config
@dataclass
class EnvConfig:
    accelerator: str = "auto"
    devices: str = "auto"
    log_every_n_steps: int = 20

    # paths
    artifacts_dir: str = "artifacts/experiments"
    input_dir_local: str = "input"
    input_dir_kaggle: str = "/kaggle/input/csiro-biomass"

    # wandb
    wandb_project: str = "csiro-biomass"
    wandb_entity: Optional[str] = None # 個人ならNone
    wandb_mode: str = "online"

@dataclass
class ExpConfig:
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

@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)

# hydra用にdefaultを設定
# YAMLで両者を合成する
cs = ConfigStore.instance()
cs.store(name="default", group="env", node=EnvConfig)
cs.store(name="default", group="exp", node=ExpConfig)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config):
    print("hydra起動")

if __name__ == "__main__":
    main()