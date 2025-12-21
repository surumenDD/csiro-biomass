"""Main training script for exp001_baseline."""
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import random
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from utils.cv import stratified_group_kfold
from utils.logger import get_logger
from utils.timing import trace
from utils.metrics import calculate_oof_metrics
from dataset import create_dataloaders
from model import BiomassModel


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def derive_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Dry_Dead_g and Dry_Clover_g from primary predictions.
    
    Args:
        df: DataFrame with columns [Dry_Total_g, GDM_g, Dry_Green_g]
        
    Returns:
        DataFrame with added Dry_Dead_g and Dry_Clover_g columns
    """
    df = df.copy()
    
    # Dry_Dead_g = max(0, Dry_Total_g - GDM_g)
    df["Dry_Dead_g"] = np.maximum(0, df["Dry_Total_g"] - df["GDM_g"])
    
    # Dry_Clover_g = max(0, GDM_g - Dry_Green_g)
    df["Dry_Clover_g"] = np.maximum(0, df["GDM_g"] - df["Dry_Green_g"])
    
    return df


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    cfg: dict,
    exp_cfg: dict,
    artifacts_dir: Path,
    logger,
):
    """Train a single fold."""
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting Fold {fold}")
    logger.info(f"{'='*50}")
    
    # Create fold directory
    fold_dir = artifacts_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    exp_name = cfg["output"]["exp_name"]
    exp_id = exp_cfg["exp_id"]
    run_name = f"{exp_name.split('_')[0].replace('exp', '')}_{exp_id}_f{fold}"
    
    wandb_logger = WandbLogger(
        name=run_name,
        project=cfg["logging"]["wandb"]["project"],
        entity=cfg["logging"]["wandb"]["entity"],
        save_dir=str(artifacts_dir / "wandb"),
        offline=cfg["logging"]["wandb"]["mode"] == "offline",
    )
    
    # Create dataloaders
    with trace("Creating dataloaders"):
        train_loader, valid_loader = create_dataloaders(
            train_df=train_df,
            valid_df=valid_df,
            img_dir=cfg["data"]["input_dir"],
            target_cols=cfg["targets"]["primary"],
            aug_cfg=exp_cfg["augmentation"],
            batch_size=exp_cfg["training"]["batch_size"],
            num_workers=exp_cfg["training"]["num_workers"],
        )
    
    logger.info(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
    
    # Calculate total steps
    total_steps = len(train_loader) * exp_cfg["training"]["epochs"]
    
    # Initialize model
    with trace("Initializing model"):
        model = BiomassModel(
            backbone=exp_cfg["model"]["backbone"],
            num_targets=exp_cfg["model"]["num_targets"],
            lr=exp_cfg["optimizer"]["lr"],
            weight_decay=exp_cfg["optimizer"]["weight_decay"],
            warmup_ratio=exp_cfg["scheduler"]["warmup_ratio"],
            loss_weights=exp_cfg["training"]["loss_weights"],
            ema_decay=exp_cfg["ema"]["decay"],
            use_ema=exp_cfg["ema"]["enabled"],
            total_steps=total_steps,
        )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=fold_dir,
            filename=exp_cfg["callbacks"]["model_checkpoint"]["filename"],
            monitor=exp_cfg["callbacks"]["model_checkpoint"]["monitor"],
            mode=exp_cfg["callbacks"]["model_checkpoint"]["mode"],
            save_top_k=exp_cfg["callbacks"]["model_checkpoint"]["save_top_k"],
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=exp_cfg["training"]["epochs"],
        logger=wandb_logger,
        callbacks=callbacks,
        precision=16 if exp_cfg["mixed_precision"] else 32,
        gradient_clip_val=exp_cfg["gradient_clip_val"],
        deterministic=True,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
    )
    
    # Train
    with trace(f"Training fold {fold}"):
        trainer.fit(model, train_loader, valid_loader)
    
    # Generate OOF predictions
    with trace(f"Generating OOF predictions for fold {fold}"):
        model.eval()
        oof_preds = []
        oof_targets = []
        oof_image_paths = []
        
        with torch.no_grad():
            for batch in valid_loader:
                images = batch["image"].to(model.device)
                targets = batch["targets"]
                image_paths = batch["image_path"]
                
                preds = model(images).cpu().numpy()
                
                oof_preds.append(preds)
                oof_targets.append(targets.numpy())
                oof_image_paths.extend(image_paths)
        
        oof_preds = np.concatenate(oof_preds)
        oof_targets = np.concatenate(oof_targets)
    
    # Create OOF DataFrame
    oof_df = pd.DataFrame({
        "image_path": oof_image_paths,
        "Dry_Total_g_pred": oof_preds[:, 0],
        "GDM_g_pred": oof_preds[:, 1],
        "Dry_Green_g_pred": oof_preds[:, 2],
        "Dry_Total_g_true": oof_targets[:, 0],
        "GDM_g_true": oof_targets[:, 1],
        "Dry_Green_g_true": oof_targets[:, 2],
    })
    
    # Derive secondary targets
    oof_df = derive_targets(oof_df.rename(columns={
        "Dry_Total_g_pred": "Dry_Total_g",
        "GDM_g_pred": "GDM_g",
        "Dry_Green_g_pred": "Dry_Green_g",
    }))
    
    oof_df = oof_df.rename(columns={
        "Dry_Total_g": "Dry_Total_g_pred",
        "GDM_g": "GDM_g_pred",
        "Dry_Green_g": "Dry_Green_g_pred",
    })
    
    oof_df["fold"] = fold
    
    # Finish wandb run
    wandb.finish()
    
    logger.info(f"Fold {fold} complete!")
    
    return oof_df


def main():
    """Main training pipeline."""
    
    # Load configs
    exp_dir = Path(__file__).parent
    with open(exp_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    with open(exp_dir / "exp" / "000.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    
    # Set seed
    set_seed(exp_cfg["seed"])
    
    # Create artifacts directory
    artifacts_base = Path(cfg["output"]["artifacts_base"])
    exp_name = cfg["output"]["exp_name"]
    exp_id = exp_cfg["exp_id"]
    artifacts_dir = artifacts_base / exp_name / exp_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log directory
    log_dir = artifacts_dir / "log"
    log_dir.mkdir(exist_ok=True)
    
    # Initialize logger
    logger = get_logger(
        file_name=f"train_{exp_name}_{exp_id}",
        file_dir=log_dir,
    )
    
    logger.info("Starting training pipeline...")
    logger.info(f"Artifacts will be saved to: {artifacts_dir}")
    
    # Load data
    with trace("Loading data"):
        train_csv_path = Path(cfg["data"]["input_dir"]) / cfg["data"]["train_csv"]
        df = pd.read_csv(train_csv_path)
        logger.info(f"Loaded {len(df)} rows from {train_csv_path}")
    
    # Create CV splits
    with trace("Creating CV splits"):
        splits = stratified_group_kfold(
            df=df,
            n_folds=cfg["cv"]["n_folds"],
            group_col=cfg["cv"]["group_col"],
            stratify_col=cfg["cv"]["stratify_col"],
            seed=cfg["cv"]["seed"],
        )
        logger.info(f"Created {len(splits)} folds")
    
    # Train each fold
    all_oof = []
    
    for fold, (train_idx, valid_idx) in enumerate(splits):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)
        
        logger.info(f"Fold {fold}: Train={len(train_df)}, Valid={len(valid_df)}")
        
        oof_df = train_fold(
            fold=fold,
            train_df=train_df,
            valid_df=valid_df,
            cfg=cfg,
            exp_cfg=exp_cfg,
            artifacts_dir=artifacts_dir,
            logger=logger,
        )
        
        all_oof.append(oof_df)
    
    # Merge OOF predictions
    with trace("Merging OOF predictions"):
        oof_full = pd.concat(all_oof, ignore_index=True)
        
        # Save OOF
        oof_dir = artifacts_dir / "OOF"
        oof_dir.mkdir(exist_ok=True)
        
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        oof_path = oof_dir / f"OOF_{exp_name}_{exp_id}_{timestamp}.csv"
        oof_full.to_csv(oof_path, index=False)
        logger.info(f"Saved OOF predictions to {oof_path}")
    
    # Convert OOF to long format for metrics
    with trace("Computing OOF metrics"):
        oof_long = []
        for target in cfg["targets"]["all"]:
            temp_df = oof_full[["image_path", "fold"]].copy()
            temp_df["target_name"] = target
            temp_df["target"] = oof_full[f"{target}_true"]
            temp_df["pred"] = oof_full[f"{target}_pred"]
            oof_long.append(temp_df)
        
        oof_long = pd.concat(oof_long, ignore_index=True)
        
        # Calculate metrics
        metrics = calculate_oof_metrics(oof_long)
        
        logger.info("\n" + "="*50)
        logger.info("Final OOF Metrics:")
        logger.info("="*50)
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")
    
    # Save metadata
    meta = {
        "exp_name": exp_name,
        "exp_id": exp_id,
        "config": cfg,
        "exp_config": exp_cfg,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "timestamp": timestamp,
    }
    
    meta_path = artifacts_dir / "meta.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(meta, f)
    
    logger.info(f"\nTraining complete! Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()