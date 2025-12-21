# 開発用 Makefile（docker compose + experiments）

## 前提
- Docker / Docker Compose が使えること
- `compose.yaml` に `dev` サービスがあること
- リポジトリがコンテナにマウントされ、コンテナ側の作業ディレクトリが `/work` であること  
  - もし違う場合は `WORKDIR` を上書きする（例: `make shell WORKDIR=/app`）

## 使い方（よく使う順）

### 1) イメージを作る
- `make build` を実行する

### 2) バックグラウンドで起動する
- `make up` を実行する

### 3) コンテナに入る
- `make shell` を実行する

### 4) GPU / CUDA を確認する（torch）
- `make gpu-check` を実行する

### 5) Python を起動する
- `make python` を実行する

### 6) ログを見る
- `make logs` を実行する

### 7) 停止する
- `make down` を実行する

### 8) 掃除する（未使用イメージも）
- `make clean` を実行する

## experiments の運用

### ディレクトリ構成（例）
- `experiments/exp000_sample/` をテンプレートとして使う
- 新規作成で `experiments/exp001_hoge/` のように増やす

### 新しい実験を作る（ホスト側でファイル作成）
- `make new NAME=hoge` を実行する
- `experiments/expNNN_hoge/` を作成する
- テンプレートから以下をコピーする
  - `run.py`
  - `config.yaml`
  - `exp/000.yaml`

### 実験を実行する（コンテナ内で実行）
- `make run EXP=1` を実行する
- `EXP` は `exp001_*` の `001` 部分を指定する
- 実行はコンテナ内で行う（`docker compose run --rm dev ...`）
- 実行対象は `experiments/exp001_*/run.py` になる

### 実験一覧を見る
- `make list` を実行する

## 変数の上書き

### サービス名を変える
- 例: `make shell SERVICE=dev`

### コンテナ内の作業ディレクトリを変える
- 例: `make run EXP=1 WORKDIR=/app`

## トラブル時の目安
- `make run` でファイルが見つからない場合は、`compose.yaml` の volume 設定と `WORKDIR` を確認する
- GPU が見えない場合は、`compose.yaml` の GPU 設定（`gpus:` や `deploy:` など）とホスト側の NVIDIA 環境を確認する
