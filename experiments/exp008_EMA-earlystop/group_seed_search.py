# group_seed_search.py
# 目的：SGKFのseed探索を「徐々に妥協」しながら行う
# 実行例：uv run python -m experiments.exp008_EMA-earlystop.group_seed_search

"""
=== TOP seeds（優先: max_miss -> sum_miss -> l1 -> size_ratio）===
all months = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]  (count=10)
 seed  max_miss  sum_miss       l1  size_ratio
15105         6        23 0.822918    1.884615
 3957         6        23 0.827460    2.042553
12834         6        23 0.829275    2.265306
 9483         6        23 0.831972    2.627907
 3474         6        23 0.838880    2.092593
 6957         6        23 0.840310    1.535714
10992         6        23 0.841154    2.136364
 5538         6        23 0.842514    2.439024
11869         6        23 0.842560    2.600000
18720         6        23 0.844131    2.466667

=== BEST seed = 15105 ===
 seed  max_miss  sum_miss       l1  size_ratio
15105         6        23 0.822918    1.884615
trn(305) -> val(52): [1, 2, 5, 6, 7, 8, 9, 10, 11] -> [2, 4, 6, 8, 9, 10]
trn(299) -> val(58): [1, 2, 4, 5, 6, 7, 8, 9, 10, 11] -> [5, 8, 9, 11]
trn(304) -> val(53): [1, 2, 4, 5, 6, 7, 8, 9, 10, 11] -> [6, 7, 8, 9, 11]
trn(261) -> val(96): [2, 4, 5, 6, 7, 8, 9, 10, 11] -> [1, 2, 5, 7, 9, 10]
trn(259) -> val(98): [1, 2, 4, 5, 6, 7, 8, 9, 10, 11] -> [5, 6, 7, 8, 9, 10]
"""
"""
=== TOP seeds（優先: max_miss -> sum_miss -> l1 -> size_ratio）===
all months = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]  (count=10)
 seed  max_miss  sum_miss       l1  size_ratio
13750         4        14 0.641545    2.272727
19740         4        14 0.653186    2.274510
 4338         4        14 0.688452    2.454545
17794         4        14 0.709143    1.965517
11802         4        14 0.714993    1.535211
 4547         4        14 0.716880    1.777778
13586         4        14 0.724839    1.283951
 1318         4        14 0.728306    2.303571
10349         4        14 0.767131    1.782609
14739         4        15 0.662461    2.960784

=== BEST seed = 13750 ===
 seed  max_miss  sum_miss       l1  size_ratio
13750         4        14 0.641545    2.272727
trn(260) -> val(97): [2, 5, 6, 7, 8, 9, 10, 11] -> [1, 2, 4, 6, 7, 8, 9]
trn(277) -> val(80): [1, 2, 4, 5, 6, 7, 8, 9, 10, 11] -> [5, 7, 8, 9, 10, 11]
trn(232) -> val(125): [1, 2, 4, 5, 6, 7, 8, 9, 10, 11] -> [2, 5, 6, 7, 8, 9, 10]
trn(302) -> val(55): [1, 2, 4, 5, 6, 7, 8, 9, 10, 11] -> [5, 6, 8, 9, 10, 11]
"""
"""
=== TOP seeds（優先: max_miss -> sum_miss -> l1 -> size_ratio）===
all months = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]  (count=10)
 seed  max_miss  sum_miss       l1  size_ratio
 2248         2         6 0.445929    1.185185
 3312         3         6 0.525925    1.178571
 2227         3         6 0.575393    1.271845
 2008         3         7 0.470739    1.878049
 4987         3         7 0.477345    1.352381
 2609         3         7 0.480732    1.435644
 4905         3         7 0.488568    1.458333
 4567         3         7 0.499643    1.236364
 3456         3         7 0.503955    1.659341
 4189         3         7 0.505249    1.430000

=== BEST seed = 2248 ===
 seed  max_miss  sum_miss       l1  size_ratio
 2248         2         6 0.445929    1.185185
trn(249) -> val(108): [2, 4, 5, 6, 7, 8, 9, 10, 11] -> [1, 2, 5, 6, 7, 8, 9, 10]
trn(229) -> val(128): [1, 2, 4, 5, 6, 7, 8, 9, 10, 11] -> [2, 5, 6, 7, 8, 9, 10, 11]
trn(236) -> val(121): [1, 2, 5, 6, 7, 8, 9, 10, 11] -> [4, 5, 6, 7, 8, 9, 10, 11]
(base) ryo52@surumePC:~/work-kaggle/csiro (feat/baseline)$ 
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .run import make_train_wide  

TRAIN_CSV = Path("input/train.csv")
N_SPLITS = 5

STRAT_COL = "State"
GROUP_COL = "Sampling_Date"

SEEDS = range(0, 5000)   
TOPK = 10                 # 上位何個表示するか
SHOW_BEST_FOLD_MONTHS = True


def add_month_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out[GROUP_COL].astype(str).apply(lambda x: int(x.split("/")[1]))
    return out


def make_folds_sgkf(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["fold"] = -1

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    X = np.zeros(len(out))
    y = out[STRAT_COL].astype(str).values
    groups = out[GROUP_COL].astype(str).values

    for fold, (_, val_idx) in enumerate(sgkf.split(X=X, y=y, groups=groups)):
        out.loc[val_idx, "fold"] = fold

    return out


def fold_missing_counts(df_with_fold: pd.DataFrame, all_months: set[int]) -> list[int]:
    # foldごとの「欠け月数」を返す
    miss = []
    for f in range(N_SPLITS):
        present = set(df_with_fold.loc[df_with_fold["fold"] == f, "month"].unique().tolist())
        miss.append(len(all_months - present))
    return miss


def month_l1_score(df_with_fold: pd.DataFrame, all_months_sorted: list[int]) -> float:
    overall = (
        df_with_fold["month"]
        .value_counts(normalize=True)
        .reindex(all_months_sorted, fill_value=0.0)
        .values
    )

    l1_sum = 0.0
    for f in range(N_SPLITS):
        sub = df_with_fold[df_with_fold["fold"] == f]
        dist = (
            sub["month"]
            .value_counts(normalize=True)
            .reindex(all_months_sorted, fill_value=0.0)
            .values
        )
        l1_sum += float(np.abs(dist - overall).sum())
    return l1_sum / N_SPLITS


def size_ratio(df_with_fold: pd.DataFrame) -> float:
    vc = df_with_fold["fold"].value_counts().reindex(range(N_SPLITS), fill_value=0).values
    return float(vc.max() / max(1, vc.min()))


def print_fold_months(df_with_fold: pd.DataFrame) -> None:
    for f in range(N_SPLITS):
        trn_months = sorted(df_with_fold.loc[df_with_fold["fold"] != f, "month"].unique().tolist())
        val_months = sorted(df_with_fold.loc[df_with_fold["fold"] == f, "month"].unique().tolist())
        print(
            f"trn({(df_with_fold['fold'] != f).sum()}) -> val({(df_with_fold['fold'] == f).sum()}): "
            f"{trn_months} -> {val_months}"
        )


def main() -> None:
    base = make_train_wide(TRAIN_CSV)
    base = add_month_col(base)

    all_months_sorted = sorted(base["month"].unique().tolist())
    all_months = set(all_months_sorted)
    m = len(all_months_sorted)

    rows = []
    for seed in SEEDS:
        df = make_folds_sgkf(base, seed=seed)

        miss_list = fold_missing_counts(df, all_months)
        max_miss = max(miss_list)              # 最悪foldの欠け月数
        sum_miss = int(sum(miss_list))         # 欠け月数の合計（全fold）
        l1 = month_l1_score(df, all_months_sorted)
        sr = size_ratio(df)

        # 「徐々に妥協」用のキー
        # 1) max_miss を最優先で小さく（どのfoldも月が揃いやすい）
        # 2) 次に sum_miss を小さく（全体として欠けが少ない）
        # 3) 次に L1 を小さく（月分布が全体に近い）
        # 4) 次に size_ratio を小さく（foldサイズが揃う）
        rows.append(
            {
                "seed": seed,
                "max_miss": max_miss,
                "sum_miss": sum_miss,
                "l1": l1,
                "size_ratio": sr,
            }
        )

    res = pd.DataFrame(rows).sort_values(
        ["max_miss", "sum_miss", "l1", "size_ratio"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    print("=== TOP seeds（優先: max_miss -> sum_miss -> l1 -> size_ratio）===")
    print(f"all months = {all_months_sorted}  (count={m})")
    print(res.head(TOPK).to_string(index=False))

    best_seed = int(res.loc[0, "seed"])
    print(f"\n=== BEST seed = {best_seed} ===")
    print(res.head(1).to_string(index=False))

    if SHOW_BEST_FOLD_MONTHS:
        best_df = make_folds_sgkf(base, seed=best_seed)
        print_fold_months(best_df)


if __name__ == "__main__":
    main()
