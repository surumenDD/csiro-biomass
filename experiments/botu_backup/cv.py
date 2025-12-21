"""Cross-validation utilities for stratified group k-fold splitting."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple


def stratified_group_kfold(
    df: pd.DataFrame,
    n_folds: int = 4,
    group_col: str = "Sampling_Date",
    stratify_col: str = "State",
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified group k-fold splits.
    
    Groups are kept together (no group appears in both train and validation),
    while maintaining the distribution of the stratification variable.
    
    Args:
        df: DataFrame containing the data
        n_folds: Number of folds
        group_col: Column to group by (e.g., Sampling_Date)
        stratify_col: Column to stratify by (e.g., State)
        seed: Random seed for reproducibility
        
    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    # Get unique groups and their stratification labels
    group_df = df.groupby(group_col)[stratify_col].agg(lambda x: x.mode()[0]).reset_index()
    group_df.columns = [group_col, stratify_col]
    
    # Create stratified k-fold on groups
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    splits = []
    for train_group_idx, val_group_idx in skf.split(group_df, group_df[stratify_col]):
        train_groups = group_df.iloc[train_group_idx][group_col].values
        val_groups = group_df.iloc[val_group_idx][group_col].values
        
        train_idx = df[df[group_col].isin(train_groups)].index.values
        val_idx = df[df[group_col].isin(val_groups)].index.values
        
        splits.append((train_idx, val_idx))
    
    return splits
