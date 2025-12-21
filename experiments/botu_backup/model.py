"""PyTorch Lightning module for biomass prediction."""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch_ema import ExponentialMovingAverage
from typing import Dict, List
import numpy as np


class BiomassModel(pl.LightningModule):
    """PyTorch Lightning module for multi-target biomass regression."""
    
    def __init__(
        self,
        backbone: str,
        num_targets: int,
        lr: float,
        weight_decay: float,
        warmup_ratio: float,
        loss_weights: Dict[str, float],
        ema_decay: float = 0.999,
        use_ema: bool = True,
        total_steps: int = None,
    ):
        """
        Args:
            backbone: timm model name (e.g., 'resnet18')
            num_targets: Number of regression targets
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_ratio: Ratio of warmup steps
            loss_weights: Dict mapping target names to loss weights
            ema_decay: EMA decay rate
            use_ema: Whether to use EMA
            total_steps: Total training steps (for scheduler)
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.total_steps = total_steps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        # Loss weights as tensor
        self.loss_weights = torch.tensor(list(loss_weights.values()), dtype=torch.float32)
        
        # Build model
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,  # Remove classification head
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_targets),
        )
        
        # Loss function
        self.criterion = nn.MSELoss(reduction='none')
        
        # EMA will be initialized in configure_optimizers
        self.ema = None
        
        # Store validation outputs for computing metrics
        self.validation_step_outputs = []
    
    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out
    
    def training_step(self, batch, batch_idx):
        images = batch["image"]
        targets = batch["targets"]
        
        # Forward pass
        preds = self(images)
        
        # Compute weighted MSE loss
        losses = self.criterion(preds, targets)  # (batch, num_targets)
        weights = self.loss_weights.to(losses.device)
        weighted_loss = (losses * weights).sum() / weights.sum()
        
        # Log
        self.log("train_loss", weighted_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return weighted_loss
    
    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        targets = batch["targets"]
        image_paths = batch["image_path"]
        
        # Forward pass
        preds = self(images)
        
        # Compute loss
        losses = self.criterion(preds, targets)
        weights = self.loss_weights.to(losses.device)
        weighted_loss = (losses * weights).sum() / weights.sum()
        
        # Store for metric calculation
        self.validation_step_outputs.append({
            "preds": preds.detach().cpu(),
            "targets": targets.detach().cpu(),
            "image_paths": image_paths,
        })
        
        # Log
        self.log("val_loss", weighted_loss, on_epoch=True, prog_bar=True)
        
        return weighted_loss
    
    def on_validation_epoch_end(self):
        # Aggregate predictions
        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])
        
        # Compute per-target MAE
        mae_per_target = (all_preds - all_targets).abs().mean(dim=0)
        
        for i, mae in enumerate(mae_per_target):
            self.log(f"val_mae_target{i}", mae, prog_bar=False)
        
        # Compute simple R² (unweighted, for monitoring)
        ss_res = ((all_targets - all_preds) ** 2).sum()
        ss_tot = ((all_targets - all_targets.mean(dim=0)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        
        self.log("val_r2", r2, prog_bar=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Initialize EMA
        if self.use_ema:
            self.ema = ExponentialMovingAverage(
                self.parameters(),
                decay=self.ema_decay,
            )
        
        # Scheduler
        if self.total_steps is not None:
            num_warmup_steps = int(self.total_steps * self.warmup_ratio)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        
        return optimizer
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        
        # Update EMA
        if self.ema is not None:
            self.ema.update()
    
    def on_train_end(self):
        # Apply EMA weights before final checkpoint
        if self.ema is not None:
            self.ema.copy_to()
