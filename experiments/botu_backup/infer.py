"""Inference script for generating test predictions."""
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
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.logger import get_logger
from utils.timing import trace
from dataset import BiomassDataset, get_transforms
from model import BiomassModel
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_model_from_checkpoint(ckpt_path: Path, cfg: dict, device: str = "cuda"):
    """Load model from checkpoint."""
    model = BiomassModel.load_from_checkpoint(
        ckpt_path,
        backbone=cfg["model"]["backbone"],
        num_targets=cfg["model"]["num_targets"],
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        warmup_ratio=cfg["scheduler"]["warmup_ratio"],
        loss_weights=cfg["training"]["loss_weights"],
        ema_decay=cfg["ema"]["decay"],
        use_ema=cfg["ema"]["enabled"],
    )
    model.to(device)
    model.eval()
    return model


def inference_single_fold(
    model,
    test_loader,
    device: str = "cuda",
):
    """Run inference with a single fold model."""
    all_preds = []
    all_image_paths = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            images = batch["image"].to(device)
            image_paths = batch["image_path"]
            
            preds = model(images).cpu().numpy()
            
            all_preds.append(preds)
            all_image_paths.extend(image_paths)
    
    all_preds = np.concatenate(all_preds)
    
    return all_preds, all_image_paths


def main():
    """Main inference pipeline."""
    
    # Load configs
    exp_dir = Path(__file__).parent
    with open(exp_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    with open(exp_dir / "exp" / "000.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    
    # Set seed
    set_seed(exp_cfg["seed"])
    
    # Setup paths
    artifacts_base = Path(cfg["output"]["artifacts_base"])
    exp_name = cfg["output"]["exp_name"]
    exp_id = exp_cfg["exp_id"]
    artifacts_dir = artifacts_base / exp_name / exp_id
    
    # Create log directory
    log_dir = artifacts_dir / "log"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize logger
    logger = get_logger(
        file_name=f"infer_{exp_name}_{exp_id}",
        file_dir=log_dir,
    )
    
    logger.info("Starting inference pipeline...")
    logger.info(f"Loading models from: {artifacts_dir}")
    
    # Load test data
    with trace("Loading test data"):
        test_csv_path = Path(cfg["data"]["input_dir"]) / cfg["data"]["test_csv"]
        test_df = pd.read_csv(test_csv_path)
        logger.info(f"Loaded {len(test_df)} test samples")
    
    # Create test dataset
    test_transform = get_transforms(exp_cfg["augmentation"], mode="valid")
    
    test_dataset = BiomassDataset(
        df=test_df,
        img_dir=cfg["data"]["input_dir"],
        target_cols=[],  # No targets for test
        transform=test_transform,
        mode="test",
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=exp_cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=exp_cfg["training"]["num_workers"],
        pin_memory=True,
    )
    
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Load models and run inference for each fold
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    all_fold_preds = []
    n_folds = cfg["cv"]["n_folds"]
    
    for fold in range(n_folds):
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing Fold {fold}")
        logger.info(f"{'='*50}")
        
        # Load checkpoint
        ckpt_path = artifacts_dir / f"fold{fold}" / f"{exp_cfg['callbacks']['model_checkpoint']['filename']}.ckpt"
        
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue
        
        with trace(f"Loading model from {ckpt_path}"):
            model = load_model_from_checkpoint(ckpt_path, exp_cfg, device)
        
        # Run inference
        fold_preds, image_paths = inference_single_fold(model, test_loader, device)
        all_fold_preds.append(fold_preds)
        
        logger.info(f"Fold {fold} predictions shape: {fold_preds.shape}")
    
    # Ensemble predictions (average across folds)
    with trace("Ensembling predictions"):
        ensemble_preds = np.mean(all_fold_preds, axis=0)
        logger.info(f"Ensemble predictions shape: {ensemble_preds.shape}")
    
    # Create predictions DataFrame
    pred_df = pd.DataFrame({
        "image_path": image_paths,
        "Dry_Total_g": ensemble_preds[:, 0],
        "GDM_g": ensemble_preds[:, 1],
        "Dry_Green_g": ensemble_preds[:, 2],
    })
    
    # Derive secondary targets
    pred_df = derive_targets(pred_df)
    
    # Convert to submission format (long format)
    with trace("Creating submission file"):
        submission_rows = []
        
        for _, row in pred_df.iterrows():
            img_path = row["image_path"]
            # Extract image ID from path (e.g., "test/ID1001187975.jpg" -> "ID1001187975")
            img_id = Path(img_path).stem
            
            for target_name in cfg["targets"]["all"]:
                sample_id = f"{img_id}__{target_name}"
                target_value = row[target_name]
                submission_rows.append({
                    "sample_id": sample_id,
                    "target": target_value,
                })
        
        submission_df = pd.DataFrame(submission_rows)
    
    # Save submission
    submission_dir = artifacts_dir / "submission"
    submission_dir.mkdir(exist_ok=True)
    
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    submission_path = submission_dir / f"SUB_{exp_name}_{exp_id}_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    
    logger.info(f"\nSubmission saved to: {submission_path}")
    logger.info(f"Submission shape: {submission_df.shape}")
    logger.info(f"Sample submission preview:")
    logger.info(f"\n{submission_df.head(10)}")
    
    logger.info("\nInference complete!")


if __name__ == "__main__":
    main()
