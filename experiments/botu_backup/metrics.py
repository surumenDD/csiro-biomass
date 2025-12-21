"""Metrics for model evaluation."""
import numpy as np
from typing import Dict


# Target weights for weighted R² calculation
TARGET_WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}


def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate globally weighted coefficient of determination (R²).
    
    Args:
        y_true: Ground truth values (N,)
        y_pred: Predicted values (N,)
        weights: Per-sample weights (N,)
        
    Returns:
        Weighted R² score
    """
    # Weighted mean
    weighted_mean = np.sum(weights * y_true) / np.sum(weights)
    
    # Residual sum of squares
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    
    # Total sum of squares
    ss_tot = np.sum(weights * (y_true - weighted_mean) ** 2)
    
    # R² calculation
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return r2


def calculate_oof_metrics(oof_df) -> Dict[str, float]:
    """
    Calculate weighted R² on OOF predictions.
    
    Args:
        oof_df: DataFrame with columns [sample_id, target_name, target, pred]
        
    Returns:
        Dictionary with global weighted R² and per-target R²
    """
    # Prepare arrays for global weighted R²
    all_y_true = []
    all_y_pred = []
    all_weights = []
    
    per_target_r2 = {}
    
    for target_name in TARGET_WEIGHTS.keys():
        target_df = oof_df[oof_df["target_name"] == target_name]
        
        if len(target_df) == 0:
            continue
            
        y_true = target_df["target"].values
        y_pred = target_df["pred"].values
        weight = TARGET_WEIGHTS[target_name]
        
        # Per-target R²
        weights_arr = np.ones(len(y_true)) * weight
        target_r2 = weighted_r2_score(y_true, y_pred, weights_arr)
        per_target_r2[target_name] = target_r2
        
        # Accumulate for global R²
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_weights.extend([weight] * len(y_true))
    
    # Global weighted R²
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_weights = np.array(all_weights)
    
    global_r2 = weighted_r2_score(all_y_true, all_y_pred, all_weights)
    
    metrics = {
        "global_weighted_r2": global_r2,
        **{f"r2_{k}": v for k, v in per_target_r2.items()},
    }
    
    return metrics
