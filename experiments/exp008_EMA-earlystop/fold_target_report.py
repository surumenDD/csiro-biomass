# fold_target_report.py
# 目的：
# - artifacts/folds_opt.csv（row_id, fold）を読み込み
# - 同じロジックで train_wide を再構築して row_id を作り、fold を merge
# - 3ターゲット（Dry_Green_g / Dry_Dead_g / Dry_Clover_g）の
#   1) 全体 vs 各fold の分布比較（ヒストグラム + Q1/Q2/Q3/Q4線）
#   2) foldごとの統計（count/mean/std/min/q25/median/q75/max）
#   3) foldサイズ、State分布、月分布（Sampling_Dateのmonth）などの確認情報
# - 画像（png）に保存し、CLIにサマリを出す
#
# 実行例：
# uv run python -m experiments.exp008_EMA-earlystop.fold_target_report \
#   --train_csv input/train.csv \
#   --fold_csv artifacts/folds_opt.csv \
#   --out_dir artifacts/fold_report
#
# 出力：
# - artifacts/fold_report/summary.txt
# - artifacts/fold_report/hist_Dry_Green_g.png
# - artifacts/fold_report/hist_Dry_Dead_g.png
# - artifacts/fold_report/hist_Dry_Clover_g.png
# - artifacts/fold_report/stats_targets.csv
# - artifacts/fold_report/state_dist.csv
# - artifacts/fold_report/month_dist.csv

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===== 列名（train.csv側） =====
SAMPLE_ID_COL = "sample_id"
TARGET_NAME_COL = "target_name"
TARGET_VALUE_COL = "target"

STATE_COL = "State"
GROUP_COL = "Sampling_Date"

GREEN_COL = "Dry_Green_g"
DEAD_COL = "Dry_Dead_g"
CLOVER_COL = "Dry_Clover_g"
TARGETS3 = [GREEN_COL, DEAD_COL, CLOVER_COL]

# make_folds_fixed_opt.py と同じ META_COLS にしてください
META_COLS = [
    "sample_id_prefix",
    "image_path",
    "Sampling_Date",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
]


def ensure_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sp = out[SAMPLE_ID_COL].astype(str).str.split("__", n=1, expand=True)
    out["sample_id_prefix"] = sp[0]
    out["sample_id_suffix"] = sp[1] if sp.shape[1] > 1 else ""
    return out


def make_train_wide(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    for c in [SAMPLE_ID_COL, TARGET_NAME_COL, TARGET_VALUE_COL]:
        if c not in df.columns:
            raise ValueError(f"train.csv に列 {c} がありません。")
    df = ensure_derived_cols(df)

    for c in META_COLS:
        if c not in df.columns:
            raise ValueError(
                f"train.csv に列 {c} がありません。META_COLSに入っています。run.py 側の生成処理と一致させてください。"
            )

    wide = (
        df.pivot_table(
            index=META_COLS,
            columns=TARGET_NAME_COL,
            values=TARGET_VALUE_COL,
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None

    for t in TARGETS3:
        if t not in wide.columns:
            raise ValueError(f"train_wide にターゲット列 {t} がありません。")

    return wide


def add_row_id_from_meta(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["row_id"] = out[META_COLS].astype(str).agg("||".join, axis=1)
    return out


def add_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 例: "2015/9/1" -> month=9
    out["month"] = out[GROUP_COL].astype(
        str).apply(lambda x: int(x.split("/")[1]))
    return out


def fold_sizes_table(df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    vc = df["fold"].value_counts().reindex(
        range(n_splits), fill_value=0).sort_index()
    mn = int(vc.min()) if len(vc) else 0
    out = pd.DataFrame({"n_val": vc.astype(int)})
    out["ratio_to_min"] = out["n_val"] / max(1, mn)
    return out


def size_ratio(df: pd.DataFrame, n_splits: int) -> float:
    vc = df["fold"].value_counts().reindex(
        range(n_splits), fill_value=0).values
    return float(vc.max() / max(1, int(vc.min())))


def state_distribution(df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    # 全体 + 各fold の State比率
    states = sorted(df[STATE_COL].astype(str).unique().tolist())
    overall = df[STATE_COL].astype(str).value_counts(
        normalize=True).reindex(states, fill_value=0.0)

    rows = []
    rows.append({"fold": "all", **overall.to_dict()})
    for f in range(n_splits):
        sub = df[df["fold"] == f]
        dist = sub[STATE_COL].astype(str).value_counts(
            normalize=True).reindex(states, fill_value=0.0)
        rows.append({"fold": f, **dist.to_dict()})
    out = pd.DataFrame(rows).set_index("fold")
    return out


def month_distribution(df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    months = sorted(df["month"].unique().tolist())
    overall = df["month"].value_counts(
        normalize=True).reindex(months, fill_value=0.0)

    rows = []
    rows.append({"fold": "all", **{m: float(overall.loc[m]) for m in months}})
    for f in range(n_splits):
        sub = df[df["fold"] == f]
        dist = sub["month"].value_counts(
            normalize=True).reindex(months, fill_value=0.0)
        rows.append({"fold": f, **{m: float(dist.loc[m]) for m in months}})
    out = pd.DataFrame(rows).set_index("fold")
    return out


def target_stats(df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    rows = []
    for name in ["all"] + list(range(n_splits)):
        sub = df if name == "all" else df[df["fold"] == name]
        for t in TARGETS3:
            x = pd.to_numeric(sub[t], errors="coerce").dropna()
            if len(x) == 0:
                rows.append(
                    {"fold": name, "target": t, "count": 0}
                )
                continue
            rows.append(
                {
                    "fold": name,
                    "target": t,
                    "count": int(x.shape[0]),
                    "mean": float(x.mean()),
                    "std": float(x.std(ddof=1)) if x.shape[0] > 1 else 0.0,
                    "min": float(x.min()),
                    "q25": float(x.quantile(0.25)),
                    "median": float(x.quantile(0.50)),
                    "q75": float(x.quantile(0.75)),
                    "max": float(x.max()),
                }
            )
    return pd.DataFrame(rows)


def l1_hist_distance(x_all: np.ndarray, x_fold: np.ndarray, bins: int = 50) -> float:
    """
    全体 vs fold のヒストグラム分布のL1距離（小さいほど近い）
    """
    lo = float(np.nanmin(x_all))
    hi = float(np.nanmax(x_all))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0
    h_all, edges = np.histogram(x_all, bins=bins, range=(lo, hi), density=True)
    h_f, _ = np.histogram(x_fold, bins=edges, density=True)
    return float(np.abs(h_f - h_all).sum() * (edges[1] - edges[0]))


def plot_hist_all_vs_folds(df: pd.DataFrame, n_splits: int, out_dir: Path, bins: int = 50) -> dict[str, dict]:
    """
    targetごとに図を保存。
    返り値：距離などのメトリクス
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict] = {}

    for t in TARGETS3:
        x_all = pd.to_numeric(df[t], errors="coerce").to_numpy()
        x_all = x_all[np.isfinite(x_all)]

        qs = np.quantile(x_all, [0.25, 0.5, 0.75]) if len(
            x_all) else [np.nan, np.nan, np.nan]

        plt.figure()
        # 全体
        plt.hist(x_all, bins=bins, density=True, alpha=0.35, label="all")

        # 各fold
        fold_l1 = {}
        for f in range(n_splits):
            xf = pd.to_numeric(
                df.loc[df["fold"] == f, t], errors="coerce").to_numpy()
            xf = xf[np.isfinite(xf)]
            plt.hist(xf, bins=bins, density=True, histtype="step",
                     linewidth=1.5, label=f"fold{f}")
            fold_l1[f] = l1_hist_distance(x_all, xf, bins=bins)

        # 四分位線（全体）
        for q in qs:
            if np.isfinite(q):
                plt.axvline(q)

        plt.title(f"{t}: all vs folds (quartile lines=all)")
        plt.legend()
        path = out_dir / f"hist_{t}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        metrics[t] = {
            "q25_all": float(qs[0]) if np.isfinite(qs[0]) else np.nan,
            "q50_all": float(qs[1]) if np.isfinite(qs[1]) else np.nan,
            "q75_all": float(qs[2]) if np.isfinite(qs[2]) else np.nan,
            "l1_hist_by_fold": fold_l1,
        }

    return metrics


def write_summary(
    out_path: Path,
    n_splits: int,
    size_tbl: pd.DataFrame,
    size_ratio_val: float,
    hist_metrics: dict[str, dict],
    stats_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append("=== fold sizes ===")
    lines.append(size_tbl.to_string())
    lines.append(f"size_ratio={size_ratio_val:.6f}")
    lines.append("")

    lines.append("=== target histogram L1 distance (all vs fold) ===")
    lines.append("小さいほど、全体分布に近いです。")
    for t in TARGETS3:
        m = hist_metrics[t]
        lines.append(
            f"[{t}] quartiles(all) q25={m['q25_all']:.6f}, q50={m['q50_all']:.6f}, q75={m['q75_all']:.6f}")
        for f in range(n_splits):
            lines.append(f"  fold{f}: l1_hist={m['l1_hist_by_fold'][f]:.6f}")
    lines.append("")

    lines.append("=== target stats (all + each fold) ===")
    # 見やすい順に
    show_cols = ["fold", "target", "count", "mean",
                 "std", "min", "q25", "median", "q75", "max"]
    lines.append(stats_df[show_cols].to_string(index=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="input/train.csv")
    ap.add_argument("--fold_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts/fold_report")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--bins", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) wide + row_id
    wide = make_train_wide(Path(args.train_csv))
    wide = add_row_id_from_meta(wide)

    # 2) fold merge
    folds = pd.read_csv(Path(args.fold_csv))
    if "row_id" not in folds.columns or "fold" not in folds.columns:
        raise ValueError("fold_csv は (row_id, fold) を含む必要があります。")

    df = wide.merge(folds[["row_id", "fold"]], on="row_id", how="left")
    if df["fold"].isna().any():
        nmiss = int(df["fold"].isna().sum())
        raise ValueError(
            f"fold が付与できていない行があります: {nmiss} 行。row_id生成ロジックが一致していません。")
    df["fold"] = df["fold"].astype(int)

    # 3) month
    df = add_month(df)

    # 4) size
    size_tbl = fold_sizes_table(df, args.n_splits)
    sr = size_ratio(df, args.n_splits)

    # 5) distributions + stats
    hist_metrics = plot_hist_all_vs_folds(
        df, args.n_splits, out_dir=out_dir, bins=args.bins)

    stats_df = target_stats(df, args.n_splits)
    stats_df.to_csv(out_dir / "stats_targets.csv", index=False)

    st_dist = state_distribution(df, args.n_splits)
    st_dist.to_csv(out_dir / "state_dist.csv")

    mo_dist = month_distribution(df, args.n_splits)
    mo_dist.to_csv(out_dir / "month_dist.csv")

    # 6) summary txt
    write_summary(
        out_path=out_dir / "summary.txt",
        n_splits=args.n_splits,
        size_tbl=size_tbl,
        size_ratio_val=sr,
        hist_metrics=hist_metrics,
        stats_df=stats_df,
    )

    # CLIにも要約を出す
    print((out_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()
