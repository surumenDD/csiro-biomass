#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
group_seed_search.py

目的:
- StratifiedGroupKFold (y=State, groups=Sampling_Date) の random_state (=seed) を変えながら、
  fold分割の「見た目（例: 月の偏り）」や「バランス」を評価して、候補seedをランキングする。

想定:
- run.py と同じ train.csv 形式
- train.csv を pivot して train_wide を作る（run.py の make_train_wide と同じ）
- $ uv run python group_seed_search.py   --train_csv ../../input/train.csv   --n_splits 5   --seeds 0:500   --topk
 10   --sort_by hybrid   --out_csv seed_search.csv
"""
"""
=== TOP seeds ===
 seed  balance_score  size_ratio  state_l1  drift_score  month_overlap  month_div  min_val  max_val   hybrid
  231       0.917643    1.265625  0.326009    -0.881612       0.235714   1.117326       64       81 0.300515
  140       1.031704    1.396825  0.317439    -0.828687       0.253333   1.082020       63       88 0.451623
  131       1.121111    1.745098  0.188006    -0.851431       0.225595   1.077026       51       89 0.525109
  172       1.136233    1.268657  0.433788    -0.771958       0.291071   1.063029       67       85 0.595862
  126       1.134854    1.517857  0.308499    -0.768399       0.283929   1.052328       56       85 0.596975
  251       1.428245    1.654545  0.386850    -1.180998       0.151905   1.332902       55       91 0.601547
  374       1.270798    1.436364  0.417217    -0.945509       0.207262   1.152771       55       79 0.608941
  182       1.366210    1.508475  0.428868    -1.063827       0.175357   1.239184       59       89 0.621531
   25       1.233074    1.440678  0.396198    -0.853864       0.229762   1.083626       59       85 0.635369
   49       1.235520    1.203125  0.516197    -0.853270       0.223810   1.077079       64       77 0.638231

=== BEST seed by hybrid ===
best_seed=231  best_metric=0.300515
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# run.py と合わせる（必要な列は pivot の index に入れる）
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


def _month_from_sampling_date(series: pd.Series, mode: str = "split1") -> pd.Series:
    """
    mode:
      - split1: 文字列を "/" でsplitして [1] を month として使う（あなたの現状ログと同じ）
      - split0: splitして [0] を month として使う
      - auto  : to_datetime を試して month を取る（失敗が多い場合は split1 にfallback）
    """
    s = series.astype(str)

    if mode == "split0":
        return s.apply(lambda x: x.split("/")[0].strip() if "/" in x else np.nan)
    if mode == "split1":
        return s.apply(lambda x: x.split("/")[1].strip() if "/" in x else np.nan)
    if mode == "auto":
        dt0 = pd.to_datetime(s, errors="coerce", dayfirst=False)
        dt1 = pd.to_datetime(s, errors="coerce", dayfirst=True)
        # 変換成功率が高い方を採用
        dt = dt0 if dt0.notna().mean() >= dt1.notna().mean() else dt1
        month = dt.dt.month
        # 失敗が多いなら split1 に戻す
        if month.isna().mean() > 0.3:
            return _month_from_sampling_date(series, mode="split1")
        return month.astype("Int64").astype(str)

    raise ValueError(f"Unknown month mode: {mode}")


def assign_folds(
    train_wide: pd.DataFrame,
    n_splits: int,
    seed: int,
    strat_col: str = "State",
    group_col: str = "Sampling_Date",
) -> pd.DataFrame:
    """
    run.py の make_folds と同じルール（SGKF: y=State, groups=Sampling_Date）で fold を付与する。
    ※ここでは探索用に print を出さない。
    """
    df = train_wide.copy()
    df["fold"] = -1

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X = np.zeros(len(df))  # API上必要
    y = df[strat_col].astype(str).values
    groups = df[group_col].astype(str).values

    for fold, (_, val_idx) in enumerate(sgkf.split(X=X, y=y, groups=groups)):
        df.loc[val_idx, "fold"] = fold

    return df


def score_balance(
    df_with_fold: pd.DataFrame,
    strat_col: str = "State",
) -> Tuple[float, float, float]:
    """
    小さいほど良い（0が理想に近い）

    - size_ratio: foldサイズの偏り（1が理想）
    - state_l1  : foldごとのState分布が全体分布からどれだけズレるか（小さいほど良い）
    """
    fold_counts = df_with_fold["fold"].value_counts().sort_index()
    size_ratio = float(fold_counts.max() / max(1, fold_counts.min()))

    overall = df_with_fold[strat_col].value_counts(normalize=True)
    l1_sum = 0.0
    for f in sorted(df_with_fold["fold"].unique()):
        dist = df_with_fold.loc[df_with_fold["fold"] == f, strat_col].value_counts(normalize=True)
        dist = dist.reindex(overall.index, fill_value=0.0)
        l1_sum += float(np.abs(dist.values - overall.values).sum())
    state_l1 = float(l1_sum / max(1, df_with_fold["fold"].nunique()))

    # 合成（重みは好み）
    balance_score = (size_ratio - 1.0) * 1.0 + state_l1 * 2.0
    return balance_score, size_ratio, state_l1


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return len(a & b) / u


def score_month_drift(
    df_with_fold: pd.DataFrame,
    month_col: str = "Sampling_Date_Month",
) -> Tuple[float, float, float]:
    """
    「月」ベースで seed の違いを見たいとき用の簡易指標

    - overlap: valid月集合どうしの平均Jaccard（小さいほど fold間で月が違う）
    - div    : valid月ヒストグラム vs 全体月ヒストグラムのL1距離（大きいほど“月の偏り”が強い）
    - drift_score = overlap - div（小さいほど良い、という形にする）
    """
    # fold -> set(month)
    fold_month_sets: Dict[int, set] = {}
    for f in sorted(df_with_fold["fold"].unique()):
        ms = df_with_fold.loc[df_with_fold["fold"] == f, month_col].dropna().astype(str).unique().tolist()
        fold_month_sets[int(f)] = set(ms)

    folds = sorted(fold_month_sets.keys())
    # overlap（平均Jaccard）
    jac = []
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            jac.append(_jaccard(fold_month_sets[folds[i]], fold_month_sets[folds[j]]))
    overlap = float(np.mean(jac)) if jac else 0.0

    # divergence（全体 vs 各fold valid の月分布 L1）
    overall = df_with_fold[month_col].astype(str).value_counts(normalize=True)
    divs = []
    for f in folds:
        dist = df_with_fold.loc[df_with_fold["fold"] == f, month_col].astype(str).value_counts(normalize=True)
        dist = dist.reindex(overall.index, fill_value=0.0)
        divs.append(float(np.abs(dist.values - overall.values).sum()))
    div = float(np.mean(divs)) if divs else 0.0

    drift_score = overlap - div
    return drift_score, overlap, div


def fold_month_log(df_with_fold: pd.DataFrame, month_col: str = "Sampling_Date_Month") -> List[str]:
    lines = []
    for f in sorted(df_with_fold["fold"].unique()):
        trn_df = df_with_fold[df_with_fold["fold"] != f]
        val_df = df_with_fold[df_with_fold["fold"] == f]
        trn_months = sorted({int(x) for x in trn_df[month_col].dropna().astype(str).unique() if str(x).isdigit()})
        val_months = sorted({int(x) for x in val_df[month_col].dropna().astype(str).unique() if str(x).isdigit()})
        lines.append(f"trn({trn_df.shape[0]}) -> val({val_df.shape[0]}): {trn_months} -> {val_months}")
    return lines


def parse_seeds(seed_spec: str) -> List[int]:
    """
    例:
      "0:200"      -> range(0,200)
      "0:200:5"    -> range(0,200,5)
      "42,43,100"  -> [42,43,100]
    """
    seed_spec = seed_spec.strip()
    if ":" in seed_spec and "," not in seed_spec:
        parts = seed_spec.split(":")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            step = 1
        elif len(parts) == 3:
            start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            raise ValueError(f"Invalid seed spec: {seed_spec}")
        return list(range(start, end, step))
    return [int(x.strip()) for x in seed_spec.split(",") if x.strip()]


@dataclass
class Row:
    seed: int
    balance_score: float
    size_ratio: float
    state_l1: float
    drift_score: float
    month_overlap: float
    month_div: float
    min_val: int
    max_val: int


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True, help="train.csv のパス")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seeds", type=str, default="0:500", help='例: "0:500" or "0:1000:5" or "42,43,44"')
    ap.add_argument("--month_mode", type=str, default="split1", choices=["split0", "split1", "auto"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--sort_by", type=str, default="hybrid", choices=["balance", "drift", "hybrid"])
    ap.add_argument("--alpha", type=float, default=0.7, help="hybrid = balance_score + alpha * drift_score")
    ap.add_argument("--out_csv", type=str, default="seed_search.csv")
    ap.add_argument("--show_top_logs", action="store_true", help="上位seedのfold月ログを表示")
    args = ap.parse_args()

    train_csv = Path(args.train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)

    train_wide = make_train_wide(train_csv)

    # 月列（探索・表示用）
    train_wide["Sampling_Date_Month"] = _month_from_sampling_date(train_wide["Sampling_Date"], mode=args.month_mode)

    # 簡易チェック（間違ったmonthになっていないか）
    months = train_wide["Sampling_Date_Month"].dropna().astype(str)
    print(f"[month_mode={args.month_mode}] unique={months.nunique()}  min={months.min()}  max={months.max()}  na={train_wide['Sampling_Date_Month'].isna().sum()}")
    print("Sampling_Date head:", train_wide["Sampling_Date"].astype(str).head(5).tolist())

    seeds = parse_seeds(args.seeds)

    try:
        from tqdm.auto import tqdm
        it = tqdm(seeds, desc="seed search")
    except Exception:
        it = seeds

    rows: List[Row] = []
    best_seed = None
    best_metric = None
    best_df = None

    for seed in it:
        df = assign_folds(train_wide, n_splits=args.n_splits, seed=seed)
        # month列は train_wide で作っているので引き継がれる想定だが、copyされるので再付与
        df["Sampling_Date_Month"] = train_wide["Sampling_Date_Month"].values

        balance_score, size_ratio, state_l1 = score_balance(df, strat_col="State")
        drift_score, overlap, div = score_month_drift(df, month_col="Sampling_Date_Month")

        vc = df["fold"].value_counts().sort_index()
        min_val = int(vc.min())
        max_val = int(vc.max())

        rows.append(
            Row(
                seed=int(seed),
                balance_score=float(balance_score),
                size_ratio=float(size_ratio),
                state_l1=float(state_l1),
                drift_score=float(drift_score),
                month_overlap=float(overlap),
                month_div=float(div),
                min_val=min_val,
                max_val=max_val,
            )
        )

        if args.sort_by == "balance":
            metric = balance_score
        elif args.sort_by == "drift":
            metric = drift_score
        else:
            metric = balance_score + args.alpha * drift_score

        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_seed = int(seed)
            best_df = df

    out = pd.DataFrame([r.__dict__ for r in rows])

    if args.sort_by == "balance":
        out = out.sort_values(["balance_score", "drift_score"], ascending=[True, True])
    elif args.sort_by == "drift":
        out = out.sort_values(["drift_score", "balance_score"], ascending=[True, True])
    else:
        out["hybrid"] = out["balance_score"] + args.alpha * out["drift_score"]
        out = out.sort_values(["hybrid", "balance_score"], ascending=[True, True])

    out.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv}")

    print("\n=== TOP seeds ===")
    head = out.head(args.topk).reset_index(drop=True)
    print(head.to_string(index=False))

    print(f"\n=== BEST seed by {args.sort_by} ===")
    print(f"best_seed={best_seed}  best_metric={best_metric:.6f}")

    if args.show_top_logs:
        print("\n=== fold month logs (TOP seeds) ===")
        for i in range(min(args.topk, len(head))):
            seed = int(head.loc[i, "seed"])
            df = assign_folds(train_wide, n_splits=args.n_splits, seed=seed)
            df["Sampling_Date_Month"] = train_wide["Sampling_Date_Month"].values
            logs = fold_month_log(df, month_col="Sampling_Date_Month")
            print(f"\n--- seed={seed} ---")
            for line in logs:
                print(line)


if __name__ == "__main__":
    main()
