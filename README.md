# Makefile（experiments + uv）

## 前提
- プロジェクトルートに `Makefile` / `pyproject.toml` がある
- テンプレート `experiments/exp000_sample/` がある（未指定時）

## よく使うコマンド

### 依存関係の同期（.venv 作成/更新）
make sync

### ロック更新（uv.lock）
make lock

### 新しい実験を作る（テンプレートからコピー）
make new NAME=hoge

### 実験を実行（uv）
make run EXP=1

追加引数を渡す:
make run EXP=1 ARGS="--help"

### exp 設定を指定して実行（uv）
make runy EXP=1 Y=000
- `Y` は `000` / `1` / `000.yaml` / `exp/000.yaml` などを指定できる

### 実験一覧
make list

### GPU / CUDA チェック（torch）
make gpu-check

## 変数の上書き（必要なときだけ）
- `EXP_ROOT`（デフォルト: `experiments`）
- `TEMPLATE`（デフォルト: `experiments/exp000_sample`）
- `UV`（デフォルト: `uv`）

例:
make run EXP=1 EXP_ROOT=experiments_alt
