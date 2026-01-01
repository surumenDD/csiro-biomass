# experiments/exp008_EMA-earlystop/make_folds_fixed_opt_ver2.py
# ver2 方針:
# - Sampling_Date で group 化（グループは移動単位）
# - State は stratified（全体分布に近づけるのではなく、fold間の偏りを減らす）
# - fold サイズは size_tol 以内（最大-最小 <= size_tol を「ハード制約」）
# - 3ターゲットは 0 / high_q / ultra_q / topk(全球上位k件の所属数) を均等化
# - bins の全体分布一致はオプション（既定は 0）

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import argparse
import math
import random

import numpy as np
import pandas as pd


# =====================
# Config / constants
# =====================
TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]

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


@dataclass
class Weights:
    w_state: float = 1.0
    w_zero: float = 1.0
    w_high: float = 1.0
    w_ultra: float = 1.0
    w_topk: float = 1.0
    w_bins: float = 0.0  # 全体分布一致（必要ならオン）


# =====================
# Utilities
# =====================
def _safe_normalize(x: np.ndarray) -> np.ndarray:
    s = float(x.sum())
    if s <= 0:
        return np.zeros_like(x, dtype=np.float64)
    return x.astype(np.float64) / s


def l1_dist(p: np.ndarray, q: np.ndarray) -> float:
    # p, q: probability vectors
    return float(np.abs(p - q).sum())


def make_wide(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv)

    sp = df["sample_id"].astype(str).str.split("__", n=1, expand=True)
    df["sample_id_prefix"] = sp[0]

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

    # 3ターゲットが揃っている前提（欠損があるなら早期に落とす）
    for t in TARGETS:
        if t not in wide.columns:
            raise ValueError(f"target column not found in wide: {t}")
        if wide[t].isna().any():
            raise ValueError(f"NaN found in target {t}. Check train.csv completeness/pivot.")
    return wide


# =====================
# Group feature building
# =====================
def build_groups(
    wide: pd.DataFrame,
    n_splits: int,
    high_q: float,
    ultra_q: float,
    topk: int,
    n_bins: int,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], Dict[str, set], Dict[str, np.ndarray]]:
    """
    return:
      gdf: group table, 1 row per Sampling_Date group
      high_thr: per-target high threshold (quantile)
      ultra_thr: per-target ultra threshold (quantile)
      topk_idx: per-target set of row indices (wide index) that belong to global topk
      bins_edges: per-target np.ndarray edges (len=n_bins+1)
    """
    wide = wide.copy()
    wide["_row_idx"] = np.arange(len(wide), dtype=np.int64)

    high_thr: Dict[str, float] = {}
    ultra_thr: Dict[str, float] = {}
    topk_idx: Dict[str, set] = {}
    bins_edges: Dict[str, np.ndarray] = {}

    for t in TARGETS:
        x = wide[t].to_numpy(dtype=np.float64)
        high_thr[t] = float(np.quantile(x, high_q))
        ultra_thr[t] = float(np.quantile(x, ultra_q))

        # global topk（同値が多い場合でも index ベースで切る）
        k = int(max(0, min(topk, len(x))))
        if k > 0:
            order = np.argsort(-x)  # descending
            topk_idx[t] = set(wide.iloc[order[:k]]["_row_idx"].tolist())
        else:
            topk_idx[t] = set()

        # bins（log1p で作ると尾が扱いやすいが、ここは素直に値で等頻度）
        # 全体分布一致を使うときだけ効くので、端を固定しておく。
        # quantile bins: n_bins 等頻度
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(x, qs)
        # 数値誤差で単調でない場合があるので修正
        edges = np.maximum.accumulate(edges)
        bins_edges[t] = edges.astype(np.float64)

    # group aggregate
    # row count
    g = wide.groupby(GROUP_COL, dropna=False)

    gdf = pd.DataFrame({GROUP_COL: g.size().index})
    gdf["n"] = g.size().to_numpy(dtype=np.int64)

    # State counts per group（カテゴリを固定）
    states = sorted(wide[STATE_COL].astype(str).fillna("NA").unique().tolist())
    state_to_i = {s: i for i, s in enumerate(states)}
    n_states = len(states)

    state_counts = np.zeros((len(gdf), n_states), dtype=np.int64)
    for gi, (key, sub) in enumerate(g):
        sc = sub[STATE_COL].astype(str).fillna("NA").value_counts()
        for s, c in sc.items():
            state_counts[gi, state_to_i[s]] = int(c)

    # targets features per group
    for t in TARGETS:
        # zero_count
        zc = g.apply(lambda sub: int((sub[t].to_numpy() == 0).sum())).to_numpy(dtype=np.int64)
        gdf[f"{t}__zero"] = zc

        # high_count, ultra_count
        ht = high_thr[t]
        ut = ultra_thr[t]
        hc = g.apply(lambda sub: int((sub[t].to_numpy(dtype=np.float64) >= ht).sum())).to_numpy(dtype=np.int64)
        uc = g.apply(lambda sub: int((sub[t].to_numpy(dtype=np.float64) >= ut).sum())).to_numpy(dtype=np.int64)
        gdf[f"{t}__high"] = hc
        gdf[f"{t}__ultra"] = uc

        # topk_count / topk_sum（尾の重さ対策）
        tkset = topk_idx[t]
        if len(tkset) > 0:
            tkc = g.apply(lambda sub: int(sub["_row_idx"].isin(tkset).sum())).to_numpy(dtype=np.int64)
            tks = g.apply(lambda sub: float(sub.loc[sub["_row_idx"].isin(tkset), t].sum())).to_numpy(dtype=np.float64)
        else:
            tkc = np.zeros(len(gdf), dtype=np.int64)
            tks = np.zeros(len(gdf), dtype=np.float64)
        gdf[f"{t}__topk_count"] = tkc
        gdf[f"{t}__topk_sum"] = tks

        # bins counts（全体分布一致のためのオプション）
        edges = bins_edges[t]
        # right=False で左閉右開（最右端だけ別扱い）
        b = np.searchsorted(edges, wide[t].to_numpy(dtype=np.float64), side="right") - 1
        b = np.clip(b, 0, len(edges) - 2)
        wide[f"{t}__bin"] = b

        # group x bin counts
        for bi in range(n_bins):
            bc = g.apply(lambda sub, _bi=bi: int((sub[f"{t}__bin"].to_numpy() == _bi).sum())).to_numpy(dtype=np.int64)
            gdf[f"{t}__bin{bi}"] = bc

    # attach state counts matrix columns
    for i, s in enumerate(states):
        gdf[f"state__{s}"] = state_counts[:, i]

    gdf = gdf.reset_index(drop=True)
    return gdf, high_thr, ultra_thr, topk_idx, bins_edges


# =====================
# Objective
# =====================
def compute_objective(
    gdf: pd.DataFrame,
    assign: np.ndarray,  # len=n_groups, values in [0..n_splits-1]
    n_splits: int,
    size_tol: int,
    weights: Weights,
    n_bins: int,
) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    ハード制約:
      max(fold_size)-min(fold_size) <= size_tol を満たさない場合は巨大ペナルティ

    スコア（小さいほど良い）:
      state: fold間の State 分布の平均L1（vs 全体ではなく、fold間のバラつき）
      zero/high/ultra/topk: fold間の率（または期待値）ばらつき
      bins: fold間の bin 分布ばらつき（オプション）
    """
    n_groups = len(gdf)
    assert assign.shape == (n_groups,)

    # fold sizes
    n = gdf["n"].to_numpy(dtype=np.int64)
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    for f in range(n_splits):
        fold_sizes[f] = int(n[assign == f].sum())
    size_range = int(fold_sizes.max() - fold_sizes.min())

    big_penalty = 0.0
    if size_range > size_tol:
        # 破ったら即死級（探索で弾くため）
        big_penalty = 1e6 + 1e4 * float(size_range - size_tol)

    # helper: fold aggregate
    def fold_sum(col: str, dtype=np.float64) -> np.ndarray:
        x = gdf[col].to_numpy(dtype=dtype)
        out = np.zeros(n_splits, dtype=np.float64)
        for f in range(n_splits):
            out[f] = float(x[assign == f].sum())
        return out

    # State distribution per fold: counts -> probs
    state_cols = [c for c in gdf.columns if c.startswith("state__")]
    state_mat = gdf[state_cols].to_numpy(dtype=np.float64)  # (n_groups, n_states)
    fold_state_counts = np.zeros((n_splits, state_mat.shape[1]), dtype=np.float64)
    for f in range(n_splits):
        fold_state_counts[f] = state_mat[assign == f].sum(axis=0)
    fold_state_probs = np.vstack([_safe_normalize(fold_state_counts[f]) for f in range(n_splits)])

    # fold間の均等性 = fold_probs の平均との差の平均L1
    state_mean_prob = fold_state_probs.mean(axis=0)
    state_l1 = np.array([l1_dist(fold_state_probs[f], state_mean_prob) for f in range(n_splits)], dtype=np.float64)
    state_mean_l1 = float(state_l1.mean())

    # target-based
    per_target_diag: Dict[str, Dict[str, float]] = {}
    score_targets = 0.0

    # fold row totals
    fold_n = fold_sizes.astype(np.float64)

    for t in TARGETS:
        # rates
        zero = fold_sum(f"{t}__zero", dtype=np.float64) / np.maximum(1.0, fold_n)
        high = fold_sum(f"{t}__high", dtype=np.float64) / np.maximum(1.0, fold_n)
        ultra = fold_sum(f"{t}__ultra", dtype=np.float64) / np.maximum(1.0, fold_n)

        # topk: 期待値で均す（count と sum の両方を使う）
        topk_count = fold_sum(f"{t}__topk_count", dtype=np.float64)
        topk_sum = fold_sum(f"{t}__topk_sum", dtype=np.float64)

        # 期待値（fold_n 比例）
        total_topk_count = float(gdf[f"{t}__topk_count"].sum())
        total_topk_sum = float(gdf[f"{t}__topk_sum"].sum())
        expected_count = total_topk_count * (fold_n / max(1.0, fold_n.sum()))
        expected_sum = total_topk_sum * (fold_n / max(1.0, fold_n.sum()))

        # fold間の均等性（平均との差の平均L1）
        def mean_l1_rate(r: np.ndarray) -> float:
            m = float(r.mean())
            return float(np.mean(np.abs(r - m)))

        zero_term = mean_l1_rate(zero)
        high_term = mean_l1_rate(high)
        ultra_term = mean_l1_rate(ultra)

        # count/sum は “期待値との差” を正規化して使う
        # （topk が 0 のときは 0）
        if total_topk_count > 0:
            topk_count_term = float(np.mean(np.abs(topk_count - expected_count)) / (total_topk_count + 1e-9))
        else:
            topk_count_term = 0.0
        if total_topk_sum > 0:
            topk_sum_term = float(np.mean(np.abs(topk_sum - expected_sum)) / (total_topk_sum + 1e-9))
        else:
            topk_sum_term = 0.0

        topk_term = 0.5 * topk_count_term + 0.5 * topk_sum_term

        # bins（必要なときだけ）
        bins_term = 0.0
        if weights.w_bins > 0:
            # foldごとの bin 分布 -> fold間平均との差
            fold_bin_probs = []
            for f in range(n_splits):
                counts = np.zeros(n_bins, dtype=np.float64)
                for bi in range(n_bins):
                    col = f"{t}__bin{bi}"
                    counts[bi] = float(gdf.loc[assign == f, col].sum())
                fold_bin_probs.append(_safe_normalize(counts))
            fold_bin_probs = np.vstack(fold_bin_probs)
            mean_prob = fold_bin_probs.mean(axis=0)
            bins_l1 = np.array([l1_dist(fold_bin_probs[f], mean_prob) for f in range(n_splits)], dtype=np.float64)
            bins_term = float(bins_l1.mean())

        per_target_diag[t] = {
            "zero_term": zero_term,
            "high_term": high_term,
            "ultra_term": ultra_term,
            "topk_term": topk_term,
            "bins_term": bins_term,
        }

        score_targets += (
            weights.w_zero * zero_term
            + weights.w_high * high_term
            + weights.w_ultra * ultra_term
            + weights.w_topk * topk_term
            + weights.w_bins * bins_term
        )

    score = (
        big_penalty
        + float(size_range) * 0.0  # size はハード制約。ソフトにしたければ係数を入れる
        + weights.w_state * state_mean_l1
        + score_targets
    )

    diag = {
        "size_range": float(size_range),
        "state_mean_l1": state_mean_l1,
        "targets_score": score_targets,
        "penalty": big_penalty,
        "objective": score,
    }
    return score, diag, per_target_diag


# =====================
# Initialization / Local search
# =====================
def greedy_init(
    gdf: pd.DataFrame,
    n_splits: int,
    size_tol: int,
    weights: Weights,
    n_bins: int,
    seed: int,
) -> np.ndarray:
    rng = random.Random(seed)

    n_groups = len(gdf)
    assign = np.full(n_groups, -1, dtype=np.int64)

    # 大きいグループから詰める
    order = np.argsort(-gdf["n"].to_numpy(dtype=np.int64))

    # 初期は完全ランダムより「均等に入れる」を優先
    fold_sizes = np.zeros(n_splits, dtype=np.int64)

    for gi in order:
        best_f = None
        best_score = None

        # fold の順序はシャッフルして局所最適を避ける
        folds = list(range(n_splits))
        rng.shuffle(folds)

        for f in folds:
            # 仮置き
            tmp = assign.copy()
            tmp[gi] = f

            # 未割当(-1)は一旦 fold0 とみなして objective を見ると変なので、
            # ここは「サイズ制約のみ」で先に判定し、スコアは軽く見る。
            # → greedy では主に size を守りつつ、state/targets を少しだけ見る。
            # 未割当は除外して objective を計算
            mask = tmp >= 0
            tmp2 = tmp.copy()
            # まだ割当がない分は 0 に寄せて計算しても良いが、誤差が出るので
            # 仮で 0 にしつつ、ペナルティが出ない範囲で選ぶ運用にする。
            tmp2[~mask] = 0

            # size check（ハード）
            n = gdf["n"].to_numpy(dtype=np.int64)
            fs = np.zeros(n_splits, dtype=np.int64)
            for ff in range(n_splits):
                fs[ff] = int(n[tmp2 == ff].sum())
            if int(fs.max() - fs.min()) > size_tol:
                continue

            sc, _, _ = compute_objective(gdf, tmp2, n_splits, size_tol, weights, n_bins)

            if best_score is None or sc < best_score:
                best_score = sc
                best_f = f

        if best_f is None:
            # size_tol が厳しすぎると詰む。ここでは最小foldへ入れて進める
            best_f = int(np.argmin(fold_sizes))

        assign[gi] = best_f
        fold_sizes[best_f] += int(gdf.loc[gi, "n"])

    return assign


def local_search(
    gdf: pd.DataFrame,
    assign: np.ndarray,
    n_splits: int,
    size_tol: int,
    weights: Weights,
    n_bins: int,
    local_steps: int,
    local_seed: int,
) -> np.ndarray:
    rng = random.Random(local_seed)

    best = assign.copy()
    best_score, _, _ = compute_objective(gdf, best, n_splits, size_tol, weights, n_bins)

    cur = best.copy()
    cur_score = best_score

    n_groups = len(gdf)
    n = gdf["n"].to_numpy(dtype=np.int64)

    # SA っぽい温度（固定スケジュール）
    t0 = 0.05
    t1 = 0.001

    def temperature(step: int) -> float:
        a = step / max(1, local_steps - 1)
        return t0 * (1 - a) + t1 * a

    for step in range(local_steps):
        # 2グループを選んで swap
        i = rng.randrange(n_groups)
        j = rng.randrange(n_groups)
        if i == j:
            continue
        fi = int(cur[i])
        fj = int(cur[j])
        if fi == fj:
            continue

        # size hard check (swap後)
        # fold sizes を毎回再計算しても n_groups=28 なので許容
        fold_sizes = np.zeros(n_splits, dtype=np.int64)
        for f in range(n_splits):
            fold_sizes[f] = int(n[cur == f].sum())

        # swap
        new_fold_sizes = fold_sizes.copy()
        new_fold_sizes[fi] -= int(n[i])
        new_fold_sizes[fj] += int(n[i])
        new_fold_sizes[fj] -= int(n[j])
        new_fold_sizes[fi] += int(n[j])

        if int(new_fold_sizes.max() - new_fold_sizes.min()) > size_tol:
            continue

        cand = cur.copy()
        cand[i], cand[j] = cand[j], cand[i]

        cand_score, _, _ = compute_objective(gdf, cand, n_splits, size_tol, weights, n_bins)

        delta = cand_score - cur_score
        if delta <= 0:
            cur = cand
            cur_score = cand_score
        else:
            temp = temperature(step)
            p = math.exp(-delta / max(1e-12, temp))
            if rng.random() < p:
                cur = cand
                cur_score = cand_score

        if cur_score < best_score:
            best = cur.copy()
            best_score = cur_score

    return best


# =====================
# Output / diagnostics
# =====================
def print_diag(
    title: str,
    gdf: pd.DataFrame,
    assign: np.ndarray,
    n_splits: int,
    size_tol: int,
    weights: Weights,
    n_bins: int,
) -> None:
    score, diag, per_t = compute_objective(gdf, assign, n_splits, size_tol, weights, n_bins)

    n = gdf["n"].to_numpy(dtype=np.int64)
    fold_sizes = [int(n[assign == f].sum()) for f in range(n_splits)]

    print(f"\n=== objective diag: {title} ===")
    print(f"objective={diag['objective']:.6f}  (penalty={diag['penalty']:.1f})")
    print(f"size_range={int(diag['size_range'])}  size_tol={size_tol}  fold_sizes={fold_sizes}")
    print(f"state_mean_l1={diag['state_mean_l1']:.6f}")
    print(f"targets_score={diag['targets_score']:.6f}")

    for t in TARGETS:
        d = per_t[t]
        print(
            f" {t}: "
            f"zero={d['zero_term']:.6f}  "
            f"high={d['high_term']:.6f}  "
            f"ultra={d['ultra_term']:.6f}  "
            f"topk={d['topk_term']:.6f}  "
            f"bins={d['bins_term']:.6f}"
        )


def save_fold_csv_ver2(
    wide: pd.DataFrame,
    assign: np.ndarray,
    out_csv: Path,
) -> None:
    """
    row_id を使わず、wide のキー列（META_COLS）+ fold を保存する。
    これで ipynb 側は merge(on=META_COLS) で良い。
    """
    # group -> fold が欲しい場合は別出力にする。ここは「各行（wide row）」に fold を付ける。
    # assign は group の割当なので、wide に展開する必要がある。
    # ここでは GROUP_COL の値に対する mapping を作って付与する。
    # ※同一 Sampling_Date は同一 fold
    # assign は gdf の順序に対応している前提なので、呼び出し側で整える。

    # 呼び出し側で wide["_group_idx"] を付けておき、それに沿って fold を付与する。
    if "_group_idx" not in wide.columns:
        raise ValueError("wide must contain _group_idx for mapping group->fold")

    out = wide[META_COLS].copy()
    out["fold"] = assign[wide["_group_idx"].to_numpy(dtype=np.int64)]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}  rows={len(out)}  cols={list(out.columns)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="input/train.csv")
    ap.add_argument("--out_csv", type=str, default="artifacts/folds_opt_ver2.csv")

    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # size hard constraint
    ap.add_argument("--size_tol", type=int, default=10)

    # thresholds
    ap.add_argument("--high_q", type=float, default=0.90)
    ap.add_argument("--ultra_q", type=float, default=0.99)
    ap.add_argument("--topk", type=int, default=12)

    # bins
    ap.add_argument("--n_bins", type=int, default=10)

    # weights
    ap.add_argument("--w_state", type=float, default=1.0)
    ap.add_argument("--w_zero", type=float, default=1.0)
    ap.add_argument("--w_high", type=float, default=1.0)
    ap.add_argument("--w_ultra", type=float, default=2.0)
    ap.add_argument("--w_topk", type=float, default=3.0)
    ap.add_argument("--w_bins", type=float, default=0.0)

    # local search
    ap.add_argument("--local_steps", type=int, default=120000)
    ap.add_argument("--local_seed", type=int, default=42)

    args = ap.parse_args()

    train_csv = Path(args.train_csv)
    out_csv = Path(args.out_csv)

    n_splits = int(args.n_splits)
    size_tol = int(args.size_tol)

    weights = Weights(
        w_state=float(args.w_state),
        w_zero=float(args.w_zero),
        w_high=float(args.w_high),
        w_ultra=float(args.w_ultra),
        w_topk=float(args.w_topk),
        w_bins=float(args.w_bins),
    )

    print("=== setup (ver2) ===")
    print(f"n_splits={n_splits}  seed={args.seed}  size_tol={size_tol}")
    print(f"high_q={args.high_q}  ultra_q={args.ultra_q}  topk={args.topk}  n_bins={args.n_bins}")
    print(
        "weights: "
        f"state={weights.w_state} zero={weights.w_zero} high={weights.w_high} "
        f"ultra={weights.w_ultra} topk={weights.w_topk} bins={weights.w_bins}"
    )

    # 1) wide
    wide = make_wide(train_csv)
    n_rows = len(wide)

    # 2) build group table
    # group index mapping
    group_keys = wide[GROUP_COL].astype(str).fillna("NA").to_numpy()
    uniq, inv = np.unique(group_keys, return_inverse=True)
    wide["_group_idx"] = inv.astype(np.int64)

    # group table in same order as uniq
    # build_groups uses groupby(GROUP_COL) -> order by key,
    # so align by constructing gdf from uniq order explicitly is safer.
    # ここでは build_groups で作った gdf を uniq と突合して並べ替える。
    gdf_raw, high_thr, ultra_thr, topk_idx, bins_edges = build_groups(
        wide,
        n_splits=n_splits,
        high_q=float(args.high_q),
        ultra_q=float(args.ultra_q),
        topk=int(args.topk),
        n_bins=int(args.n_bins),
    )

    # reorder gdf to match uniq order (stringified)
    gdf_raw["_k"] = gdf_raw[GROUP_COL].astype(str).fillna("NA")
    key_to_pos = {k: i for i, k in enumerate(uniq.tolist())}
    pos = gdf_raw["_k"].map(key_to_pos).to_numpy(dtype=np.int64)
    gdf = gdf_raw.iloc[np.argsort(pos)].drop(columns=["_k"]).reset_index(drop=True)

    n_groups = len(gdf)
    print(f"n_groups={n_groups}  n_rows={n_rows}")

    # 3) init + local
    assign0 = greedy_init(gdf, n_splits, size_tol, weights, int(args.n_bins), seed=int(args.seed))
    print_diag("before_local", gdf, assign0, n_splits, size_tol, weights, int(args.n_bins))

    assign = local_search(
        gdf=gdf,
        assign=assign0,
        n_splits=n_splits,
        size_tol=size_tol,
        weights=weights,
        n_bins=int(args.n_bins),
        local_steps=int(args.local_steps),
        local_seed=int(args.local_seed),
    )
    print_diag("after_local", gdf, assign, n_splits, size_tol, weights, int(args.n_bins))

    # 4) save (ver2)
    save_fold_csv_ver2(wide, assign, out_csv)


if __name__ == "__main__":
    main()
