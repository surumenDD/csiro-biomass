#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
split_seed 探索スクリプト（学習なし）
- train.csv -> train_wide（画像1枚=1行）に変換
- StratifiedKFold の random_state(seed) を複数試す
- foldサイズ / State分布 / ターゲット平均のズレ を合成スコア化して最小のseedを選ぶ

元コードの以下を再利用できるように、同じロジックで最小構成にしてある：
- make_train_wide / make_folds / _score_split_balance / make_folds_with_seed_search
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# ===== あなたのコードと同じ列定義 =====
PRED3_COLS = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g"]

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
    df[["sample_id_prefix", "sample_id_suffix"]] = df["sample_id"].str.split("__", expand=True)

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


def make_folds(
    train_wide: pd.DataFrame,
    n_splits: int,
    seed: int,
    strat_col: str = "State",
    group_col: str = "Sampling_Date",
) -> pd.DataFrame:
    df = train_wide.copy()
    df["fold"] = -1

    # 元コードと同じ：StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X = np.zeros(len(df))  # API上必要
    y = df[strat_col].astype(str).values

    for fold, (_, val_idx) in enumerate(skf.split(X=X, y=y)):
        df.loc[val_idx, "fold"] = fold

    return df


def _score_split_balance(
    df_with_fold: pd.DataFrame,
    strat_col: str = "State",
    pred3_cols: Optional[list[str]] = None,
) -> float:
    """
    小さいほど良い（0が理想）
    - foldサイズの偏り
    - strat_col(State) 分布の偏り
    - 3ターゲット平均の偏り（任意）
    """
    fold_counts = df_with_fold["fold"].value_counts().sort_index()
    size_ratio = float(fold_counts.max() / max(1, fold_counts.min()))  # 1が理想

    # State分布のズレ（L1距離の平均）
    overall = df_with_fold[strat_col].value_counts(normalize=True)
    l1_sum = 0.0
    for f in sorted(df_with_fold["fold"].unique()):
        dist = (
            df_with_fold.loc[df_with_fold["fold"] == f, strat_col]
            .value_counts(normalize=True)
            .reindex(overall.index, fill_value=0.0)
        )
        l1_sum += float(np.abs(dist.values - overall.values).sum())
    state_l1 = float(l1_sum / max(1, df_with_fold["fold"].nunique()))

    # ターゲット平均のズレ（任意）
    target_pen = 0.0
    if pred3_cols is not None and all(c in df_with_fold.columns for c in pred3_cols):
        global_mean = df_with_fold[pred3_cols].mean()
        per = df_with_fold.groupby("fold")[pred3_cols].mean()
        target_pen = float(np.abs(per - global_mean).mean().mean())

    # 合成（元コードと同じ重み）
    score = (size_ratio - 1.0) * 1.0 + state_l1 * 2.0 + target_pen * 1.0
    return float(score)


def make_folds_with_seed_search(
    train_wide: pd.DataFrame,
    n_splits: int,
    seed_candidates: list[int],
    strat_col: str = "State",
    group_col: str = "Sampling_Date",
    pred3_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    best_seed: Optional[int] = None
    best_df: Optional[pd.DataFrame] = None
    best_score: Optional[float] = None
    rows: list[dict] = []

    for seed in seed_candidates:
        df = make_folds(
            train_wide=train_wide,
            n_splits=n_splits,
            seed=seed,
            strat_col=strat_col,
            group_col=group_col,
        )
        score = _score_split_balance(df, strat_col=strat_col, pred3_cols=pred3_cols)
        rows.append({"seed": int(seed), "score": float(score)})

        if best_score is None or score < best_score:
            best_score = float(score)
            best_seed = int(seed)
            best_df = df

    if best_df is None or best_seed is None:
        raise RuntimeError("seed search failed: no candidate produced a split")

    score_df = pd.DataFrame(rows).sort_values("score", ascending=True).reset_index(drop=True)
    return best_df, best_seed, score_df


def _summary_table(best_df: pd.DataFrame, strat_col: str, pred3_cols: Optional[list[str]]) -> pd.DataFrame:
    # foldサイズ
    sizes = best_df["fold"].value_counts().sort_index().rename("n").to_frame()
    # strat分布の上位だけ（確認用）
    top_states = best_df[strat_col].value_counts().head(10).index.tolist()
    for s in top_states:
        sizes[f"{strat_col}={s}"] = (best_df[strat_col].astype(str) == str(s)).groupby(best_df["fold"]).mean().values
    if pred3_cols is not None and all(c in best_df.columns for c in pred3_cols):
        for c in pred3_cols:
            sizes[f"mean({c})"] = best_df.groupby("fold")[c].mean().values
    return sizes.reset_index().rename(columns={"index": "fold"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="input/train.csv")
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--seed_min", type=int, default=0)
    ap.add_argument("--seed_max", type=int, default=200, help="この値は含めない（rangeと同じ）")
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--strat_col", type=str, default="State")
    ap.add_argument("--group_col", type=str, default="Sampling_Date")
    ap.add_argument("--no_target_pen", action="store_true", help="ターゲット平均ペナルティを使わない")
    ap.add_argument("--out_dir", type=str, default="artifacts/seed_search")
    ap.add_argument("--save_best_folds", action="store_true")
    args = ap.parse_args()

    train_csv = Path(args.train_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.exists():
        raise FileNotFoundError(f"train_csv not found: {train_csv}")

    train_wide = make_train_wide(train_csv)
    # 安全：型揃え（元コードと同じ）
    train_wide["image_path"] = train_wide["image_path"].astype(str)
    train_wide["sample_id_prefix"] = train_wide["sample_id_prefix"].astype(str)

    # ここは「探索」なので seed を広めに見る
    seed_candidates = list(range(int(args.seed_min), int(args.seed_max)))

    pred3_cols = None if args.no_target_pen else PRED3_COLS

    best_df, best_seed, score_df = make_folds_with_seed_search(
        train_wide=train_wide,
        n_splits=int(args.n_splits),
        seed_candidates=seed_candidates,
        strat_col=str(args.strat_col),
        group_col=str(args.group_col),
        pred3_cols=pred3_cols,
    )

    # 保存
    score_path = out_dir / "seed_scores_4.csv"
    score_df.to_csv(score_path, index=False)

    # 結果出力
    print("===== split_seed search (no training) =====")
    print(f"train_csv: {train_csv}")
    print(f"n_splits: {args.n_splits}")
    print(f"seed range: [{args.seed_min}, {args.seed_max})  candidates={len(seed_candidates)}")
    print(f"use_target_penalty: {not args.no_target_pen}")
    print("")
    print(f"BEST split_seed = {best_seed}   score={float(score_df.iloc[0]['score']):.6f}")
    print("")
    print("Top seeds:")
    print(score_df.head(int(args.top_k)).to_string(index=False))

    print("")
    print("Best split summary (per fold):")
    summ = _summary_table(best_df, strat_col=str(args.strat_col), pred3_cols=pred3_cols)
    print(summ.to_string(index=False))

    if args.save_best_folds:
        best_path = out_dir / f"train_wide_with_folds_seed{best_seed}.csv"
        best_df.to_csv(best_path, index=False)
        print("")
        print(f"Saved best folds table: {best_path}")

    print("")
    print(f"Saved seed score ranking: {score_path}")


if __name__ == "__main__":
    main()
