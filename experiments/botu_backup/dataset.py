"""Dataset and DataLoader for biomass prediction."""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class BiomassDataset(Dataset):
    """Dataset for pasture biomass image prediction."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        target_cols: List[str],
        transform=None,
        mode: str = "train",
    ):
        """
        Args:
            df: DataFrame with image_path and target columns
            img_dir: Base directory containing images
            target_cols: List of target column names to predict
            transform: Albumentations transform
            mode: 'train' or 'valid' or 'test'
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.target_cols = target_cols
        self.transform = transform
        self.mode = mode
        
        # For train/valid, we need unique images (df has 5 rows per image)
        if mode in ["train", "valid"]:
            self.image_ids = df["image_path"].unique()
        else:
            self.image_ids = df["image_path"].values
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_ids[idx]
        
        # Load image
        full_path = self.img_dir / img_path
        image = cv2.imread(str(full_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        result = {"image": image, "image_path": img_path}
        
        # Add targets if in train/valid mode
        if self.mode in ["train", "valid"]:
            # Get the row corresponding to this image
            img_df = self.df[self.df["image_path"] == img_path]
            
            # Extract targets (pivot to get all targets in one row)
            targets = []
            for target_col in self.target_cols:
                target_row = img_df[img_df["target_name"] == target_col]
                if len(target_row) > 0:
                    targets.append(target_row["target"].values[0])
                else:
                    targets.append(0.0)  # Fallback
            
            result["targets"] = torch.tensor(targets, dtype=torch.float32)
        
        return result


def get_transforms(cfg: dict, mode: str = "train"):
    """
    Create albumentations transforms.
    
    Args:
        cfg: Configuration dict with augmentation settings
        mode: 'train' or 'valid'
        
    Returns:
        Albumentations Compose transform
    """
    aug_cfg = cfg.get(mode, {})
    
    transforms_list = []
    
    # Add augmentations based on config
    for aug_name, aug_params in aug_cfg.items():
        if aug_name == "HorizontalFlip":
            transforms_list.append(A.HorizontalFlip(**aug_params))
        elif aug_name == "VerticalFlip":
            transforms_list.append(A.VerticalFlip(**aug_params))
        elif aug_name == "RandomBrightnessContrast":
            transforms_list.append(A.RandomBrightnessContrast(**aug_params))
        elif aug_name == "Normalize":
            transforms_list.append(A.Normalize(**aug_params))
    
    # Always add ToTensorV2 at the end
    transforms_list.append(ToTensorV2())
    
    return A.Compose(transforms_list)


def create_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    img_dir: str,
    target_cols: List[str],
    aug_cfg: dict,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_df: Training DataFrame
        valid_df: Validation DataFrame
        img_dir: Image directory
        target_cols: Target column names
        aug_cfg: Augmentation config
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        (train_loader, valid_loader)
    """
    train_transform = get_transforms(aug_cfg, mode="train")
    valid_transform = get_transforms(aug_cfg, mode="valid")
    
    train_dataset = BiomassDataset(
        df=train_df,
        img_dir=img_dir,
        target_cols=target_cols,
        transform=train_transform,
        mode="train",
    )
    
    valid_dataset = BiomassDataset(
        df=valid_df,
        img_dir=img_dir,
        target_cols=target_cols,
        transform=valid_transform,
        mode="valid",
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, valid_loader
