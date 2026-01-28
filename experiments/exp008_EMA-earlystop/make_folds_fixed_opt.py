# make_folds_fixed_opt.py (rebuilt)
# 方針:
# - group: Sampling_Date（同日付は同fold）
# - stratified (main): State
# - hard constraint: max(fold_size) - min(fold_size) <= size_tol (default 20)
# - stratified (secondary): targets (Dry_Green_g, Dry_Dead_g, Dry_Clover_g) の
#     - zero率
#     - high率（上位 high_q 分位以上）
#   を fold 間で揃える
# - optional: 3ターゲットの全体分布（quantile bins）も揃える

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import argparse
import numpy as np
import pandas as pd


# =====================
# Config
# =====================
META_COLS = [
    "sample_id_prefix",
    "image_path",
    "Sampling_Date",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
]
GROUP_COL = "Sampling_Date"
STATE_COL = "State"

TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]


# =====================
# IO: train.csv (long) -> wide
# =====================
def make_train_wide(train_csv: Path) -> pd.DataFrame:
    """
    train.csv の想定:
      - sample_id: "<prefix>__<suffix>"
      - target_name: e.g. Dry_Green_g
      - target: float
      - image_path, Sampling_Date, State, ...（meta列）
    """
    df = pd.read_csv(train_csv)

    if "sample_id" not in df.columns:
        raise ValueError("train.csv に sample_id 列がありません。")
    if "target_name" not in df.columns:
        raise ValueError("train.csv に target_name 列がありません。")
    if "target" not in df.columns:
        raise ValueError("train.csv に target 列がありません。")

    df[["sample_id_prefix", "sample_id_suffix"]] = df["sample_id"].astype(str).str.split("__", expand=True)

    # META_COLS が無いと困るのでチェック
    for c in META_COLS:
        if c not in df.columns:
            raise ValueError(f"train.csv に必要な列 {c} がありません。列名を確認してください。")

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

    # ターゲット列の存在チェック（無い場合はエラーにする）
    for t in TARGETS:
        if t not in wide.columns:
            raise ValueError(f"pivot後のwideに {t} 列がありません。target_name を確認してください。")

    return wide


# =====================
# Group-level feature table
# =====================
@dataclass
class Thresholds:
    high_q: float
    high_thr: Dict[str, float]


def compute_thresholds(df: pd.DataFrame, high_q: float) -> Thresholds:
    high_thr: Dict[str, float] = {}
    for t in TARGETS:
        x = pd.to_numeric(df[t], errors="coerce").fillna(0.0).astype(float)
        high_thr[t] = float(x.quantile(high_q))
    return Thresholds(high_q=high_q, high_thr=high_thr)


def build_group_table(
    df: pd.DataFrame,
    thr: Thresholds,
    n_bins: int = 0,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Sampling_Date ごとに集約したテーブルを作る。
    - n: サンプル数
    - state one-hot counts
    - targetごとの zero/high counts
    - (optional) targetごとの quantile bin counts
    """
    w = df.copy()

    # state
    w["_state"] = w[STATE_COL].astype(str)

    # zero/high flags
    for t in TARGETS:
        x = pd.to_numeric(w[t], errors="coerce").fillna(0.0).astype(float)
        w[f"{t}__is_zero"] = (x == 0.0).astype(int)
        w[f"{t}__is_high"] = (x >= thr.high_thr[t]).astype(int)

    # optional: bins
    bin_cols: List[str] = []
    if n_bins and n_bins > 1:
        for t in TARGETS:
            x = pd.to_numeric(w[t], errors="coerce").fillna(0.0).astype(float)
            # qcutは同値が多いと bins が潰れるので duplicates="drop"
            b = pd.qcut(x, q=n_bins, labels=False, duplicates="drop")
            # それでも NaN が出ることがあるので埋める（最小bin扱い）
            b = b.fillna(0).astype(int)
            col = f"{t}__qbin"
            w[col] = b
            bin_cols.append(col)

    # group size
    g = w.groupby(GROUP_COL, dropna=False)
    gdf = g.size().rename("n").reset_index()

    # state counts
    state_counts = (
        w.assign(_one=1)
        .pivot_table(index=GROUP_COL, columns="_state", values="_one", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    state_cols = [c for c in state_counts.columns if c != GROUP_COL]
    state_counts = state_counts.rename(columns={c: f"state__{c}" for c in state_cols})
    state_cols = [f"state__{c}" for c in state_cols]

    gdf = gdf.merge(state_counts, on=GROUP_COL, how="left")

    # zero/high counts
    extreme_cols: List[str] = []
    for t in TARGETS:
        for kind in ["is_zero", "is_high"]:
            col = f"{t}__{kind}"
            extreme_cols.append(col)
            tmp = w.groupby(GROUP_COL)[col].sum().rename(col).reset_index()
            gdf = gdf.merge(tmp, on=GROUP_COL, how="left")

    # bin counts (optional)
    tbin_cols: List[str] = []
    if bin_cols:
        for bcol in bin_cols:
            # bcol の値ごとに count を列にする（例: Dry_Green_g__qbin=0..）
            # ただし列名を固定長にしないと比較が面倒なので、bin id を付けた列にする
            tmp = (
                w.assign(_one=1)
                .pivot_table(index=GROUP_COL, columns=bcol, values="_one", aggfunc="sum", fill_value=0)
                .reset_index()
            )
            # columns: GROUP_COL, 0,1,2,...
            # -> GROUP_COL, bcol__bin0, bcol__bin1 ...
            rename_map = {}
            for c in tmp.columns:
                if c == GROUP_COL:
                    continue
                rename_map[c] = f"{bcol}__bin{int(c)}"
                tbin_cols.append(rename_map[c])
            tmp = tmp.rename(columns=rename_map)
            gdf = gdf.merge(tmp, on=GROUP_COL, how="left")

    # fillna
    for c in state_cols + extreme_cols + tbin_cols:
        if c in gdf.columns:
            gdf[c] = gdf[c].fillna(0).astype(int)

    return gdf, state_cols, (extreme_cols + tbin_cols)


# =====================
# Objective
# =====================
def _l1_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.abs(p - q).sum())


def _safe_prob(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    # Laplace smoothing
    x = counts.astype(float) + alpha
    s = float(x.sum())
    if s <= 0:
        return np.zeros_like(x, dtype=float)
    return x / s


@dataclass
class Weights:
    w_state: float
    w_zero: float
    w_high: float
    w_bins: float  # optional


@dataclass
class ObjDiag:
    objective: float
    size_range: int
    state_mean: float
    extreme_mean: float
    bins_mean: float


def compute_objective(
    fold_to_groups: List[List[int]],
    gdf: pd.DataFrame,
    state_cols: List[str],
    feat_cols: List[str],
    n_splits: int,
    size_tol: int,
    weights: Weights,
    alpha_smooth: float = 1.0,
) -> ObjDiag:
    """
    ハード制約:
      - size_range <= size_tol を強烈に罰する（実質禁止）
    ソフト:
      - state分布（one-hot counts）を全体に近づける
      - extreme/ bins を全体に近づける（ここでは “全体 vs fold” の L1）
    """
    # fold sizes
    fold_sizes = []
    for f in range(n_splits):
        idx = fold_to_groups[f]
        fold_sizes.append(int(gdf.loc[idx, "n"].sum()))
    size_range = int(max(fold_sizes) - min(fold_sizes))

    # hard penalty
    hard_pen = 0.0
    if size_range > size_tol:
        # 超えた分だけ、かなり強い罰
        hard_pen = (size_range - size_tol) * 1000.0

    # overall state distribution
    overall_state = gdf[state_cols].sum(axis=0).values.astype(float)
    overall_state_p = _safe_prob(overall_state, alpha=alpha_smooth)

    # overall feats
    overall_feat = gdf[feat_cols].sum(axis=0).values.astype(float)
    overall_feat_p = _safe_prob(overall_feat, alpha=alpha_smooth)

    state_l1s = []
    feat_l1s = []

    # state / feat for each fold
    for f in range(n_splits):
        idx = fold_to_groups[f]
        sub_state = gdf.loc[idx, state_cols].sum(axis=0).values.astype(float)
        sub_feat = gdf.loc[idx, feat_cols].sum(axis=0).values.astype(float)

        sp = _safe_prob(sub_state, alpha=alpha_smooth)
        fp = _safe_prob(sub_feat, alpha=alpha_smooth)

        state_l1s.append(_l1_dist(sp, overall_state_p))
        feat_l1s.append(_l1_dist(fp, overall_feat_p))

    state_mean = float(np.mean(state_l1s))
    feat_mean = float(np.mean(feat_l1s))

    # feat_cols の内訳（zero/high と bins）で分けて平均も出す（診断用）
    zero_high_cols = [c for c in feat_cols if c.endswith("__is_zero") or c.endswith("__is_high")]
    bins_cols = [c for c in feat_cols if "__qbin__bin" in c]

    extreme_mean = 0.0
    bins_mean = 0.0
    if zero_high_cols:
        overall_e = gdf[zero_high_cols].sum(axis=0).values.astype(float)
        overall_e_p = _safe_prob(overall_e, alpha=alpha_smooth)
        e_l1s = []
        for f in range(n_splits):
            idx = fold_to_groups[f]
            sub_e = gdf.loc[idx, zero_high_cols].sum(axis=0).values.astype(float)
            e_l1s.append(_l1_dist(_safe_prob(sub_e, alpha=alpha_smooth), overall_e_p))
        extreme_mean = float(np.mean(e_l1s))

    if bins_cols:
        overall_b = gdf[bins_cols].sum(axis=0).values.astype(float)
        overall_b_p = _safe_prob(overall_b, alpha=alpha_smooth)
        b_l1s = []
        for f in range(n_splits):
            idx = fold_to_groups[f]
            sub_b = gdf.loc[idx, bins_cols].sum(axis=0).values.astype(float)
            b_l1s.append(_l1_dist(_safe_prob(sub_b, alpha=alpha_smooth), overall_b_p))
        bins_mean = float(np.mean(b_l1s))

    # objective: hard + soft
    # feat_mean は “zero/high + bins” 全体のL1平均
    # ここで w_zero/w_high を個別に分けたい場合は、feat_cols を分割して別々に測るが、
    # まずは仕様を単純にする（zero/high を強め、bins はオプション扱い）
    obj = hard_pen
    obj += weights.w_state * state_mean

    # zero/high を強める（bins は w_bins 次第）
    obj += weights.w_zero * extreme_mean  # zero/highまとめてここに入る
    obj += weights.w_bins * bins_mean

    # 注: w_high を別に使いたい場合は is_high と is_zero を分けて測る設計に変える
    # ただし “0 と大きい値を重視” なら、extreme_mean を十分に重くすれば実質達成できる

    return ObjDiag(
        objective=float(obj),
        size_range=size_range,
        state_mean=state_mean,
        extreme_mean=extreme_mean,
        bins_mean=bins_mean,
    )


# =====================
# Construct initial folds (greedy)
# =====================
def greedy_init(
    gdf: pd.DataFrame,
    state_cols: List[str],
    feat_cols: List[str],
    n_splits: int,
    size_tol: int,
    weights: Weights,
    seed: int,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)

    # group indices sorted by size desc (harder first)
    order = np.argsort(-gdf["n"].values.astype(int))
    order = order.tolist()

    # fold assignments: list of group indices
    folds: List[List[int]] = [[] for _ in range(n_splits)]

    # random tie-break
    def _choose_best_fold(gi: int) -> int:
        candidates = list(range(n_splits))
        rng.shuffle(candidates)

        best_f = candidates[0]
        best_obj = None

        for f in candidates:
            # try assign gi to fold f
            tmp = [lst.copy() for lst in folds]
            tmp[f].append(gi)

            diag = compute_objective(
                tmp, gdf, state_cols, feat_cols,
                n_splits=n_splits, size_tol=size_tol, weights=weights
            )

            if best_obj is None or diag.objective < best_obj:
                best_obj = diag.objective
                best_f = f
        return best_f

    for gi in order:
        f = _choose_best_fold(gi)
        folds[f].append(gi)

    return folds


# =====================
# Local search (swap / move) with hard size constraint
# =====================
def _fold_sizes(folds: List[List[int]], gdf: pd.DataFrame) -> List[int]:
    out = []
    for idx in folds:
        out.append(int(gdf.loc[idx, "n"].sum()) if len(idx) else 0)
    return out


def local_search(
    folds: List[List[int]],
    gdf: pd.DataFrame,
    state_cols: List[str],
    feat_cols: List[str],
    n_splits: int,
    size_tol: int,
    weights: Weights,
    steps: int,
    seed: int,
) -> Tuple[List[List[int]], ObjDiag]:
    rng = np.random.default_rng(seed)

    best = [lst.copy() for lst in folds]
    best_diag = compute_objective(best, gdf, state_cols, feat_cols, n_splits, size_tol, weights)

    for _ in range(steps):
        # pick two folds
        a, b = rng.integers(0, n_splits, size=2)
        if a == b:
            continue
        if len(best[a]) == 0 or len(best[b]) == 0:
            continue

        # propose swap (one group from each)
        ga = best[a][rng.integers(0, len(best[a]))]
        gb = best[b][rng.integers(0, len(best[b]))]

        cand = [lst.copy() for lst in best]
        # swap
        cand[a].remove(ga)
        cand[b].remove(gb)
        cand[a].append(gb)
        cand[b].append(ga)

        # quick hard size check (skip expensive objective if broken)
        sizes = _fold_sizes(cand, gdf)
        if max(sizes) - min(sizes) > size_tol:
            continue

        diag = compute_objective(cand, gdf, state_cols, feat_cols, n_splits, size_tol, weights)
        if diag.objective < best_diag.objective:
            best = cand
            best_diag = diag

    return best, best_diag


# =====================
# Emit fold csv (sample_id_prefix -> fold)
# =====================
def assign_folds_to_rows(df: pd.DataFrame, gdf: pd.DataFrame, folds: List[List[int]]) -> pd.DataFrame:
    """
    gdf は Sampling_Date 集約なので、foldは Sampling_Date によって決まる。
    df（wide）に fold を付与して返す。
    """
    # group value
    group_vals = gdf[GROUP_COL].tolist()

    g2fold: Dict[str, int] = {}
    for f, idxs in enumerate(folds):
        for gi in idxs:
            g2fold[str(group_vals[gi])] = int(f)

    out = df.copy()
    out["fold"] = out[GROUP_COL].astype(str).map(g2fold).astype(int)
    return out


def print_fold_summary(df_with_fold: pd.DataFrame, n_splits: int) -> None:
    vc = df_with_fold["fold"].value_counts().reindex(range(n_splits), fill_value=0)
    sizes = vc.values.astype(int)
    print("=== fold sizes (row count) ===")
    for f in range(n_splits):
        print(f"fold={f}: n_rows={int(sizes[f])}")
    print(f"size_range_rows={int(sizes.max() - sizes.min())}")


# =====================
# CLI
# =====================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="input/train.csv")
    p.add_argument("--out_csv", type=str, required=True)

    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # hard constraint
    p.add_argument("--size_tol", type=int, default=20, help="max(fold_size)-min(fold_size) <= size_tol を強制")

    # extremes
    p.add_argument("--high_q", type=float, default=0.90, help="大きい値の閾値を上位 high_q 分位で作る")

    # optional bins
    p.add_argument("--n_bins", type=int, default=0, help="0なら無効。>1でquantile bin分布も揃える")

    # weights
    p.add_argument("--w_state", type=float, default=1.0)
    p.add_argument("--w_extreme", type=float, default=4.0, help="zero/high をまとめて強める")
    p.add_argument("--w_bins", type=float, default=0.0, help="bin分布の重み（0で無効）")

    # local search
    p.add_argument("--local_steps", type=int, default=120000)
    p.add_argument("--local_seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    n_splits = int(args.n_splits)
    size_tol = int(args.size_tol)

    # load
    df = make_train_wide(Path(args.train_csv))

    # thresholds for high
    thr = compute_thresholds(df, high_q=float(args.high_q))

    # group features
    gdf, state_cols, feat_cols = build_group_table(df, thr, n_bins=int(args.n_bins))

    weights = Weights(
        w_state=float(args.w_state),
        w_zero=float(args.w_extreme),  # zero/highまとめてここに入れる
        w_high=0.0,
        w_bins=float(args.w_bins),
    )

    print("=== setup ===")
    print(f"n_splits={n_splits} size_tol={size_tol} seed={int(args.seed)}")
    print(f"high_q={thr.high_q} high_thr={{"
          + ", ".join([f"{k}:{v:.6f}" for k, v in thr.high_thr.items()]) + "}}")
    print(f"weights: state={weights.w_state} extreme={weights.w_zero} bins={weights.w_bins}")
    print(f"n_groups={len(gdf)} n_rows={len(df)}")

    # init
    folds0 = greedy_init(gdf, state_cols, feat_cols, n_splits, size_tol, weights, seed=int(args.seed))
    diag0 = compute_objective(folds0, gdf, state_cols, feat_cols, n_splits, size_tol, weights)
    sizes0 = _fold_sizes(folds0, gdf)

    print("\n=== objective diag: before_local ===")
    print(f"objective={diag0.objective:.6f}")
    print(f"size_range={diag0.size_range}  fold_sizes={sizes0}")
    print(f"state_mean_l1={diag0.state_mean:.6f}")
    print(f"extreme_mean_l1={diag0.extreme_mean:.6f}")
    if int(args.n_bins) and int(args.n_bins) > 1:
        print(f"bins_mean_l1={diag0.bins_mean:.6f}")

    # local search (hard size constraint enforced by skipping broken moves)
    folds1, diag1 = local_search(
        folds0, gdf, state_cols, feat_cols,
        n_splits=n_splits, size_tol=size_tol, weights=weights,
        steps=int(args.local_steps), seed=int(args.local_seed),
    )
    sizes1 = _fold_sizes(folds1, gdf)

    print("\n=== objective diag: after_local ===")
    print(f"objective={diag1.objective:.6f}")
    print(f"size_range={diag1.size_range}  fold_sizes={sizes1}")
    print(f"state_mean_l1={diag1.state_mean:.6f}")
    print(f"extreme_mean_l1={diag1.extreme_mean:.6f}")
    if int(args.n_bins) and int(args.n_bins) > 1:
        print(f"bins_mean_l1={diag1.bins_mean:.6f}")

    # assign to rows
    df_out = assign_folds_to_rows(df, gdf, folds1)

    # final sanity
    print()
    print_fold_summary(df_out, n_splits=n_splits)

    # save (fold割当は sample_id_prefix 単位で使う前提)
    # ただし pipeline 側で wide を期待している可能性があるので wide+fold を保存する
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
