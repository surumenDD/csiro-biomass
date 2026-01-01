# optuna_search_folds_by_training.py
# 目的:
#  1) make_folds_fixed_opt_ver2.py で fold を生成する
#  2) 生成した fold を使って実際に学習を回す
#  3) foldごとのCVスコアの「ばらつき」を小さくする（同じくらいにする）
#
# 返す目的関数（最大化）:
#   mean(score_f) - alpha * std(score_f)
# さらに「全foldが低スコアで揃う」を避けるために、mean が低いと罰則を入れられる。

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna


# =========================
# 設定
# =========================
@dataclass
class Settings:
    # ここをあなたの環境に合わせて変更する
    project_root: Path = Path(__file__).resolve().parents[2]  # このスクリプトの場所
    train_csv: Path = Path("input/train.csv")

    # make_folds の実体（ファイルパス）
    make_folds_py: Path = Path("experiments/exp008_EMA-earlystop/make_folds_fixed_opt_ver2.py")

    # 学習コマンド（あなたの実装に合わせてここだけ変える）
    # - {fold} と {folds_csv} を埋め込む
    # - 例: "uv run python -m experiments.exp008_EMA-earlystop.run exp=004 fold={fold} folds_csv={folds_csv}"
    train_cmd_template: str = (
        "uv run python -m experiments.exp011_original_5fold.run "
        "exp=005 fold={fold} folds_csv={folds_csv}"
    )

    # fold数
    n_splits: int = 5

    # make_folds 側の seed（trial 間比較を安定させるため、まず固定が無難）
    seed: int = 5555
    local_seed: int = 5555

    # local search の試行回数（重いのでまず固定推奨）
    local_steps: int = 120000

    # 目的関数（揃え具合）の係数
    alpha_std: float = 0.5  # std をどれだけ嫌うか（大きいほど「揃える」寄り）
    mean_floor: Optional[float] = None  # 例: 0.30 など。Noneなら罰則なし
    mean_floor_penalty: float = 3.0     # mean が floor を下回ったときの罰則の強さ

    # Optuna
    n_trials: int = 30
    timeout_sec: Optional[int] = None
    sampler_seed: int = 20260101

    # 出力先
    work_dir: Path = Path("artifacts/optuna_folds_train")
    save_best_folds_csv: Path = Path("artifacts/folds_optuna_best.csv")

    # 学習ログから拾うスコア文字列（あなたの実装に合わせて追加・変更）
    score_regex_candidates: Tuple[str, ...] = (
        r"val_weighted_r2\s*=\s*([0-9.+-eE]+)",
        r"val_wr2\s*=\s*([0-9.+-eE]+)",
        r"weighted_r2\s*=\s*([0-9.+-eE]+)",
        r"wr2\s*=\s*([0-9.+-eE]+)",
        r"R2\s*=\s*([0-9.+-eE]+)",
    )

    # 学習が JSON を出している場合、その場所を探すための候補（任意）
    # 例: trainer 側で metrics.json を出しているなら一致させる
    metrics_json_name_candidates: Tuple[str, ...] = ("metrics.json", "cv_metrics.json")


# =========================
# ユーティリティ
# =========================
def _run_command(command: str, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> str:
    """shell でコマンド実行し、stdout+stderr を返す。失敗したら例外。"""
    p = subprocess.run(
        command,
        shell=True,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = p.stdout
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (code={p.returncode}).\nCOMMAND:\n{command}\n\nOUTPUT:\n{out}")
    return out


def _extract_last_float_by_regex(text: str, patterns: Tuple[str, ...]) -> Optional[float]:
    """ログ文字列から、指定パターンの最後の一致を float として返す。"""
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            try:
                return float(matches[-1])
            except Exception:
                continue
    return None


def _find_metrics_json(trial_dir: Path, candidates: Tuple[str, ...]) -> Optional[Path]:
    """trial_dir 配下を軽く探して metrics json を見つける。"""
    for name in candidates:
        p = trial_dir / name
        if p.exists():
            return p
    # 深い探索は遅いので、2階層までにする
    for depth in (1, 2):
        for p in trial_dir.glob("/".join(["*"] * depth) + "/*.json"):
            if p.name in candidates:
                return p
    return None


def _read_score_from_metrics_json(p: Path) -> Optional[float]:
    """metrics json に fold score がある前提でスコアを読む。存在しなければ None。"""
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None

    # よくあるキーの候補
    keys = ["val_weighted_r2", "weighted_r2", "wr2", "score", "metric", "best_wr2"]
    for k in keys:
        if k in data and isinstance(data[k], (int, float)):
            return float(data[k])

    # {"metrics": {"val_weighted_r2": ...}} 形式
    if isinstance(data.get("metrics"), dict):
        for k in keys:
            if k in data["metrics"] and isinstance(data["metrics"][k], (int, float)):
                return float(data["metrics"][k])

    return None


# =========================
# make_folds 実行
# =========================
def run_make_folds(st: Settings, out_csv: Path, params: Dict[str, float]) -> None:
    """
    make_folds_fixed_opt_ver2.py を実行して out_csv を作る。
    params は探索するハイパラ。
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"{sys.executable} {st.make_folds_py} "
        f"--train_csv {st.train_csv} "
        f"--out_csv {out_csv} "
        f"--n_splits {st.n_splits} "
        f"--seed {st.seed} "
        f"--local_seed {st.local_seed} "
        f"--local_steps {st.local_steps} "
        f"--size_tol {int(params['size_tol'])} "
        f"--high_q {params['high_q']:.6f} "
        f"--ultra_q {params['ultra_q']:.6f} "
        f"--topk {int(params['topk'])} "
        f"--w_state {params['w_state']:.6f} "
        f"--w_zero {params['w_zero']:.6f} "
        f"--w_high {params['w_high']:.6f} "
        f"--w_ultra {params['w_ultra']:.6f} "
        f"--w_topk {params['w_topk']:.6f} "
        f"--w_bins {params['w_bins']:.6f} "
    )
    _run_command(cmd, cwd=st.project_root)

    if not out_csv.exists():
        raise RuntimeError(f"folds csv was not created: {out_csv}")


# =========================
# 学習（fold単位）
# =========================
def run_train_one_fold(st: Settings, folds_csv: Path, fold: int, trial_dir: Path) -> float:
    """
    1 fold の学習を実行して、CV指標（例: val_weighted_r2）を float で返す。
    - まず stdout ログから拾う
    - 次に trial_dir の metrics json を探して拾う（任意）
    """
    trial_dir.mkdir(parents=True, exist_ok=True)

    folds_csv_abs = folds_csv.resolve()
    cmd = st.train_cmd_template.format(fold=fold, folds_csv=str(folds_csv))
    out = _run_command(cmd, cwd=None)

    # 1) stdout から読む
    score = _extract_last_float_by_regex(out, st.score_regex_candidates)
    if score is not None and math.isfinite(score):
        (trial_dir / f"train_fold{fold}.log").write_text(out)
        return float(score)

    # 2) metrics json を探して読む（あなたの学習が json を出す場合に有効）
    (trial_dir / f"train_fold{fold}.log").write_text(out)
    mj = _find_metrics_json(trial_dir, st.metrics_json_name_candidates)
    if mj is not None:
        s2 = _read_score_from_metrics_json(mj)
        if s2 is not None and math.isfinite(s2):
            return float(s2)

    raise RuntimeError(
        "Score was not found.\n"
        "あなたの学習ログにスコアが出ていないか、正規表現が合っていません。\n"
        "Settings.score_regex_candidates を学習ログに合わせて修正してください。"
    )


# =========================
# Optuna 目的関数
# =========================
def make_params(trial: optuna.Trial) -> Dict[str, float]:
    size_tol = trial.suggest_int("size_tol", 0, 30)

    high_q = trial.suggest_float("high_q", 0.80, 0.95)
    ultra_q = trial.suggest_float("ultra_q", max(0.90, high_q + 0.01), 0.999)

    topk = trial.suggest_int("topk", 5, 60)

    w_state = trial.suggest_float("w_state", 0.0, 5.0)
    w_zero = trial.suggest_float("w_zero", 0.0, 5.0)
    w_high = trial.suggest_float("w_high", 0.0, 5.0)
    w_ultra = trial.suggest_float("w_ultra", 0.0, 8.0)
    w_topk = trial.suggest_float("w_topk", 0.0, 10.0)
    w_bins = trial.suggest_float("w_bins", 0.0, 2.0)

    return {
        "size_tol": float(size_tol),
        "high_q": float(high_q),
        "ultra_q": float(ultra_q),
        "topk": float(topk),
        "w_state": float(w_state),
        "w_zero": float(w_zero),
        "w_high": float(w_high),
        "w_ultra": float(w_ultra),
        "w_topk": float(w_topk),
        "w_bins": float(w_bins),
    }


def objective_factory(st: Settings):
    def objective(trial: optuna.Trial) -> float:
        params = make_params(trial)

        trial_dir = st.work_dir / f"trial_{trial.number:05d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        folds_csv = trial_dir / "folds.csv"

        # 1) fold を作る
        run_make_folds(st, folds_csv, params)

        # 2) 実学習で fold スコアを取る
        scores: List[float] = []
        for f in range(st.n_splits):
            fold_dir = trial_dir / f"fold_{f}"
            score_f = run_train_one_fold(st, folds_csv, f, fold_dir)
            scores.append(score_f)

            # pruner 用に途中経過を report
            # step は fold index にする（foldが進むごとに情報が増える）
            cur_mean = float(np.mean(scores))
            trial.report(cur_mean, step=f)
            if trial.should_prune():
                raise optuna.TrialPruned()

        arr = np.array(scores, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        mn = float(arr.min())

        # 3) 目的関数を作る（最大化）
        #    mean を上げつつ std を下げる
        value = mean - st.alpha_std * std

        # mean が低すぎる「悪く揃う」を避けたい場合
        if st.mean_floor is not None and mean < st.mean_floor:
            value -= st.mean_floor_penalty * (st.mean_floor - mean)

        # trial に保存（後で見る用）
        trial.set_user_attr("scores", scores)
        trial.set_user_attr("mean", mean)
        trial.set_user_attr("std", std)
        trial.set_user_attr("min", mn)
        trial.set_user_attr("params", params)

        # ログも保存
        (trial_dir / "result.json").write_text(
            json.dumps(
                {"scores": scores, "mean": mean, "std": std, "min": mn, "value": value, "params": params},
                indent=2,
                ensure_ascii=False,
            )
        )

        return float(value)

    return objective


# =========================
# メイン
# =========================
def main() -> None:
    st = Settings()
    st.work_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=st.sampler_seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)  # foldが2つ進むまで刈らない

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    obj = objective_factory(st)

    study.optimize(obj, n_trials=st.n_trials, timeout=st.timeout_sec)

    best = study.best_trial
    print("best_value:", best.value)
    print("best_params:", best.params)
    print("best_scores:", best.user_attrs.get("scores"))
    print("best_mean:", best.user_attrs.get("mean"), "best_std:", best.user_attrs.get("std"))

    # best の folds.csv を保存
    best_trial_dir = st.work_dir / f"trial_{best.number:05d}"
    best_folds = best_trial_dir / "folds.csv"
    st.save_best_folds_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_folds, st.save_best_folds_csv)
    print("saved_best_folds_csv:", st.save_best_folds_csv)

    # study の全結果も保存
    out_study = st.work_dir / "study_best.json"
    out_study.write_text(
        json.dumps(
            {
                "best_value": best.value,
                "best_params": best.params,
                "best_scores": best.user_attrs.get("scores"),
                "best_mean": best.user_attrs.get("mean"),
                "best_std": best.user_attrs.get("std"),
                "best_trial_number": best.number,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print("saved:", out_study)


if __name__ == "__main__":
    main()
