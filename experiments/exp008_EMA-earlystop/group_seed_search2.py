# group_seed_search.py
# 目的：
# - StratifiedGroupKFold を使い、groups=Sampling_Date は固定（同日付は同fold）
# - stratify は以下を連結して作る（文字列ラベル）：
#   1) State
#   2) Dry_Clover_g の Q1/Q4/O
#   3) Dry_Green_g の Q4/Q2Q3/O
#   4) Dry_Dead_g  の Q4/Q2Q3/O
# - seed を探索し、
#   1) size_ratio（foldサイズ均等）を最優先で最小化
#   2) 3ターゲット分布のズレ（targets_l1）を次に最小化（分位ビン分布L1）
#   3) stratifyラベル分布のズレ（strat_l1）で同点を崩す
#
# 実行：
# uv run python -m experiments.exp008_EMA-earlystop.group_seed_search

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .run import make_train_wide

# ===== 警告を表示しない（必要なものだけ消す）=====
warnings.filterwarnings(
    "ignore",
    message=r"The least populated class in y has only",
    category=UserWarning,
    module=r"sklearn\.model_selection\._split",
)

TRAIN_CSV = Path("input/train.csv")
N_SPLITS = 5

STATE_COL = "State"
GROUP_COL = "Sampling_Date"

CLOVER_COL = "Dry_Clover_g"
GREEN_COL = "Dry_Green_g"
DEAD_COL = "Dry_Dead_g"

# 分布を揃えたい3ターゲット
TARGETS3 = [GREEN_COL, DEAD_COL, CLOVER_COL]

SEEDS = range(0, 5000)
TOPK = 10
SHOW_BEST_INFO = True


# ===== stratify 用ラベル作成 =====
def add_clover_q_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clover を Q1 / Q4 / O にする
      Q1: x <= q25
      Q4: x >  q75
      O : それ以外（Q2+Q3）
    """
    out = df.copy()
    if CLOVER_COL not in out.columns:
        raise ValueError(f"{CLOVER_COL} が df にありません。make_train_wide の出力列を確認してください。")

    x = pd.to_numeric(out[CLOVER_COL], errors="coerce")
    q25 = float(x.quantile(0.25))
    q75 = float(x.quantile(0.75))

    lab = np.full(len(out), "O", dtype=object)
    lab[x <= q25] = "Q1"
    lab[x > q75] = "Q4"
    out["clover_q"] = lab
    return out


def add_green_dead_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Green/Dead を Q4 / Q2Q3 / O にする（ユーザー指定）
      Q4  : x >  q75
      Q2Q3: q25 < x <= q75
      O   : x <= q25  （= Q1）
    """
    out = df.copy()

    for col, new_col in [(GREEN_COL, "green_q"), (DEAD_COL, "dead_q")]:
        if col not in out.columns:
            raise ValueError(f"{col} が df にありません。make_train_wide の出力列を確認してください。")

        x = pd.to_numeric(out[col], errors="coerce")
        q25 = float(x.quantile(0.25))
        q75 = float(x.quantile(0.75))

        lab = np.full(len(out), "Q2Q3", dtype=object)
        lab[x <= q25] = "O"
        lab[x > q75] = "Q4"
        out[new_col] = lab

    return out


def make_strat_y(df: pd.DataFrame) -> np.ndarray:
    # y = State + Clover + Green + Dead
    return (
        df[STATE_COL].astype(str)
        + "_" + df["clover_q"].astype(str)
        + "_" + df["green_q"].astype(str)
        + "_" + df["dead_q"].astype(str)
    ).values


# ===== fold 作成 =====
def make_folds_sgkf(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["fold"] = -1

    X = np.zeros(len(out))
    y = make_strat_y(out)
    groups = out[GROUP_COL].astype(str).values

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(sgkf.split(X=X, y=y, groups=groups)):
        out.loc[val_idx, "fold"] = fold
    return out


# ===== 評価指標 =====
def size_ratio(df_with_fold: pd.DataFrame) -> float:
    vc = df_with_fold["fold"].value_counts().reindex(range(N_SPLITS), fill_value=0).values
    return float(vc.max() / max(1, vc.min()))


def strat_l1(df_with_fold: pd.DataFrame) -> float:
    """
    foldごとの stratifyラベル分布のズレ（L1）
    """
    y_all = make_strat_y(df_with_fold)
    overall = pd.Series(y_all).value_counts(normalize=True)

    l1_sum = 0.0
    for f in range(N_SPLITS):
        sub = df_with_fold[df_with_fold["fold"] == f]
        y_sub = make_strat_y(sub)
        dist = pd.Series(y_sub).value_counts(normalize=True).reindex(overall.index, fill_value=0.0)
        l1_sum += float(np.abs(dist.values - overall.values).sum())
    return l1_sum / N_SPLITS


def add_target_quartile_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    targets_l1 用：各ターゲットを Q1..Q4 の4値(1..4)にする
    """
    out = df.copy()
    for col in TARGETS3:
        if col not in out.columns:
            raise ValueError(f"{col} が df にありません。make_train_wide の出力列を確認してください。")

        x = pd.to_numeric(out[col], errors="coerce")
        q25 = float(x.quantile(0.25))
        q50 = float(x.quantile(0.50))
        q75 = float(x.quantile(0.75))

        b = np.full(len(out), 3, dtype=int)
        b[x <= q25] = 1
        b[(x > q25) & (x <= q50)] = 2
        b[(x > q50) & (x <= q75)] = 3
        b[x > q75] = 4
        out[f"{col}__qbin"] = b
    return out


def targets_l1(df_with_fold: pd.DataFrame) -> float:
    """
    3ターゲットそれぞれについて、foldごとの(Q1..Q4)分布と全体分布のL1を平均。
    最後に3ターゲット平均を返す。
    """
    score_sum = 0.0
    for col in TARGETS3:
        bin_col = f"{col}__qbin"
        overall = (
            df_with_fold[bin_col]
            .value_counts(normalize=True)
            .reindex([1, 2, 3, 4], fill_value=0.0)
            .values
        )

        l1_sum = 0.0
        for f in range(N_SPLITS):
            sub = df_with_fold[df_with_fold["fold"] == f]
            dist = (
                sub[bin_col]
                .value_counts(normalize=True)
                .reindex([1, 2, 3, 4], fill_value=0.0)
                .values
            )
            l1_sum += float(np.abs(dist - overall).sum())

        score_sum += l1_sum / N_SPLITS

    return score_sum / len(TARGETS3)


def print_fold_sizes(df_with_fold: pd.DataFrame) -> None:
    vc = df_with_fold["fold"].value_counts().reindex(range(N_SPLITS), fill_value=0)
    for f in range(N_SPLITS):
        print(f"fold={f}: n_val={int(vc.loc[f])}")
    print(f"size_ratio={size_ratio(df_with_fold):.6f}")


def main() -> None:
    base = make_train_wide(TRAIN_CSV)
    base = add_clover_q_label(base)
    base = add_green_dead_labels(base)
    base = add_target_quartile_bins(base)

    rows = []
    for seed in SEEDS:
        df = make_folds_sgkf(base, seed=seed)

        sr = size_ratio(df)
        tl1 = targets_l1(df)
        sl1 = strat_l1(df)

        rows.append({"seed": seed, "size_ratio": sr, "targets_l1": tl1, "strat_l1": sl1})

    res = pd.DataFrame(rows).sort_values(
        ["size_ratio", "targets_l1", "strat_l1"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    print("=== TOP seeds（優先: size_ratio -> targets_l1 -> strat_l1）===")
    print(res.head(TOPK).to_string(index=False))

    best_seed = int(res.loc[0, "seed"])
    print(f"\n=== BEST seed = {best_seed} ===")
    print(res.head(1).to_string(index=False))

    if SHOW_BEST_INFO:
        best_df = make_folds_sgkf(base, seed=best_seed)
        print_fold_sizes(best_df)


if __name__ == "__main__":
    main()
