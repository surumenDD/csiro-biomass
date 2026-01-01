# fold_target_compare.py
# 目的：
# - 複数seedについて同じCV分割（State+clover_q+green_q+dead_qで層化、Sampling_Dateでgroup）を作る
# - 各seedの各foldにおける 3ターゲット（Green/Dead/Clover）の分布を、同じレイアウトで比較しやすく描画する
# - 追加で、seedごとの size_ratio / targets_l1 / strat_l1 を計算して一覧表示・CSV保存する
#
# 実行例:
# uv run python -m experiments.exp008_EMA-earlystop.fold_target_compare \
#   --seeds 3056,3850,4327,1020,863,3747 \
#   --out_dir artifacts/fold_compare
#
# 出力:
# - compare_boxplot.png
# - compare_hist.png
# - seed_metrics.csv

from __future__ import annotations

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .run import make_train_wide
except Exception:
    from run import make_train_wide  # type: ignore


# ===== sklearn の警告を消す（必要なものだけ）=====
warnings.filterwarnings(
    "ignore",
    message=r"The least populated class in y has only",
    category=UserWarning,
    module=r"sklearn\.model_selection\._split",
)

TRAIN_CSV_DEFAULT = "input/train.csv"
N_SPLITS_DEFAULT = 5

STATE_COL = "State"
GROUP_COL = "Sampling_Date"

CLOVER_COL = "Dry_Clover_g"
GREEN_COL = "Dry_Green_g"
DEAD_COL = "Dry_Dead_g"

TARGETS3 = [GREEN_COL, DEAD_COL, CLOVER_COL]


def parse_seeds(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ===== stratify 用ラベル（seed_search と一致させる）=====
def add_clover_q_label(df: pd.DataFrame) -> pd.DataFrame:
    # Clover: Q1 / Q4 / O
    out = df.copy()
    x = pd.to_numeric(out[CLOVER_COL], errors="coerce")
    q25 = float(x.quantile(0.25))
    q75 = float(x.quantile(0.75))
    lab = np.full(len(out), "O", dtype=object)
    lab[x <= q25] = "Q1"
    lab[x > q75] = "Q4"
    out["clover_q"] = lab
    return out


def add_green_dead_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Green/Dead: Q4 / Q2Q3 / O（O=下位=Q1）
    out = df.copy()
    for col, new_col in [(GREEN_COL, "green_q"), (DEAD_COL, "dead_q")]:
        x = pd.to_numeric(out[col], errors="coerce")
        q25 = float(x.quantile(0.25))
        q75 = float(x.quantile(0.75))
        lab = np.full(len(out), "Q2Q3", dtype=object)
        lab[x <= q25] = "O"
        lab[x > q75] = "Q4"
        out[new_col] = lab
    return out


def make_strat_y(df: pd.DataFrame) -> np.ndarray:
    return (
        df[STATE_COL].astype(str)
        + "_" + df["clover_q"].astype(str)
        + "_" + df["green_q"].astype(str)
        + "_" + df["dead_q"].astype(str)
    ).values


def make_folds(df: pd.DataFrame, seed: int, n_splits: int) -> pd.DataFrame:
    out = df.copy()
    out["fold"] = -1

    X = np.zeros(len(out))
    y = make_strat_y(out)
    groups = out[GROUP_COL].astype(str).values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(sgkf.split(X=X, y=y, groups=groups)):
        out.loc[val_idx, "fold"] = fold

    return out


# ===== 指標（seed_search と一致）=====
def size_ratio(df_with_fold: pd.DataFrame, n_splits: int) -> float:
    vc = df_with_fold["fold"].value_counts().reindex(range(n_splits), fill_value=0).values
    return float(vc.max() / max(1, vc.min()))


def add_target_quartile_bins(df: pd.DataFrame) -> pd.DataFrame:
    # targets_l1 用：各ターゲットを Q1..Q4 の4値(1..4)
    out = df.copy()
    for col in TARGETS3:
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


def targets_l1(df_with_fold: pd.DataFrame, n_splits: int) -> float:
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
        for f in range(n_splits):
            sub = df_with_fold[df_with_fold["fold"] == f]
            dist = (
                sub[bin_col]
                .value_counts(normalize=True)
                .reindex([1, 2, 3, 4], fill_value=0.0)
                .values
            )
            l1_sum += float(np.abs(dist - overall).sum())

        score_sum += l1_sum / n_splits

    return score_sum / len(TARGETS3)


def strat_l1(df_with_fold: pd.DataFrame, n_splits: int) -> float:
    y_all = make_strat_y(df_with_fold)
    overall = pd.Series(y_all).value_counts(normalize=True)

    l1_sum = 0.0
    for f in range(n_splits):
        sub = df_with_fold[df_with_fold["fold"] == f]
        y_sub = make_strat_y(sub)
        dist = pd.Series(y_sub).value_counts(normalize=True).reindex(overall.index, fill_value=0.0)
        l1_sum += float(np.abs(dist.values - overall.values).sum())

    return l1_sum / n_splits


# ===== 描画用 =====
def compute_global_hist_bins(base: pd.DataFrame, col: str, n_bins: int = 30) -> tuple[np.ndarray, float, float]:
    x = pd.to_numeric(base[col], errors="coerce").dropna().values
    lo, hi = np.quantile(x, [0.01, 0.99])
    bins = np.linspace(lo, hi, n_bins + 1)
    return bins, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default=TRAIN_CSV_DEFAULT)
    ap.add_argument("--seeds", type=str, required=True)
    ap.add_argument("--n_splits", type=int, default=N_SPLITS_DEFAULT)
    ap.add_argument("--out_dir", type=str, default="artifacts/fold_compare")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    if len(seeds) == 0:
        raise ValueError("--seeds が空です。例: --seeds 3056,3850,4327")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = make_train_wide(Path(args.train_csv))
    base = add_clover_q_label(base)
    base = add_green_dead_labels(base)
    base = add_target_quartile_bins(base)  # 指標にも使う

    # hist bins をseed間で揃える
    hist_cfg = {col: compute_global_hist_bins(base, col, n_bins=30) for col in TARGETS3}

    # ---- seedごとの指標を計算して保存 ----
    metrics_rows = []
    dfs_by_seed: dict[int, pd.DataFrame] = {}

    for seed in seeds:
        df = make_folds(base, seed=seed, n_splits=args.n_splits)
        dfs_by_seed[seed] = df

        metrics_rows.append(
            {
                "seed": seed,
                "size_ratio": size_ratio(df, args.n_splits),
                "targets_l1": targets_l1(df, args.n_splits),
                "strat_l1": strat_l1(df, args.n_splits),
                "fold_sizes": ",".join(
                    str(int(v)) for v in df["fold"].value_counts().reindex(range(args.n_splits), fill_value=0).sort_index().values
                ),
            }
        )

    metrics = pd.DataFrame(metrics_rows).sort_values(["size_ratio", "targets_l1", "strat_l1"]).reset_index(drop=True)
    metrics.to_csv(out_dir / "seed_metrics.csv", index=False)
    print("=== seed metrics ===")
    print(metrics.to_string(index=False))
    print(f"[saved] {out_dir}/seed_metrics.csv")

    # ==========
    # 図1: boxplot（行=seed、列=ターゲット）
    # ==========
    n_rows = len(seeds)
    n_cols = len(TARGETS3)

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes1 = np.array([axes1])

    for r, seed in enumerate(seeds):
        df = dfs_by_seed[seed]
        for c, col in enumerate(TARGETS3):
            ax = axes1[r, c]
            data = [
                pd.to_numeric(df.loc[df["fold"] == f, col], errors="coerce").dropna().values
                for f in range(args.n_splits)
            ]
            ax.boxplot(data, labels=[str(f) for f in range(args.n_splits)], showfliers=False)

            if r == 0:
                ax.set_title(col)
            if c == 0:
                ax.set_ylabel(f"seed={seed}")
            ax.set_xlabel("fold")

    fig1.suptitle("Boxplot by fold (rows=seed, cols=target)", y=1.02)
    fig1.tight_layout()
    fig1.savefig(out_dir / "compare_boxplot.png", dpi=200)
    plt.close(fig1)
    print(f"[saved] {out_dir}/compare_boxplot.png")

    # ==========
    # 図2: hist（行=seed、列=ターゲット）
    # ==========
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes2 = np.array([axes2])

    for r, seed in enumerate(seeds):
        df = dfs_by_seed[seed]
        for c, col in enumerate(TARGETS3):
            ax = axes2[r, c]
            bins, lo, hi = hist_cfg[col]

            for f in range(args.n_splits):
                xs = pd.to_numeric(df.loc[df["fold"] == f, col], errors="coerce").dropna().values
                xs = xs[(xs >= lo) & (xs <= hi)]
                ax.hist(xs, bins=bins, histtype="step", linewidth=1, density=True, label=f"f{f}")

            if r == 0:
                ax.set_title(col)
            if c == 0:
                ax.set_ylabel(f"seed={seed}\ndensity")
            ax.set_xlabel(col)
            if r == 0 and c == n_cols - 1:
                ax.legend(ncols=min(args.n_splits, 5), fontsize=8, loc="upper left")

    fig2.suptitle("Histogram by fold (rows=seed, cols=target) [clipped 1%-99%]", y=1.02)
    fig2.tight_layout()
    fig2.savefig(out_dir / "compare_hist.png", dpi=200)
    plt.close(fig2)
    print(f"[saved] {out_dir}/compare_hist.png")


if __name__ == "__main__":
    main()
