# Makefile (experiments + uv)
# 使い方:
#   make new  NAME=hoge
#   make run  EXP=1 ARGS="--help"
#   make runy EXP=1 Y=000 ARGS=""
#   make list
#   make sync

SHELL := /bin/bash

# ====== 設定（未指定ならこの値） ======
EXP_ROOT ?= experiments
TEMPLATE ?= $(EXP_ROOT)/exp000_sample

UV ?= uv
PY ?= $(UV) run python

# Makefile がある場所をプロジェクトルートとして扱う
PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: help new run runy infer list sync lock gpu-check

help:
	@echo "Targets:"
	@echo "  new   NAME=...   新しい実験を作成"
	@echo "  run   EXP=...    実験を実行 (例: make run EXP=1 ARGS='')"
	@echo "  runy  EXP=... Y=...   exp設定を指定して実行 (例: make runy EXP=1 Y=000)"
	@echo "  list            実験一覧"
	@echo "  sync            uv sync"
	@echo "  lock            uv lock"

# =========================
# uv の依存関係を同期
# =========================
sync:
	@$(UV) sync

lock:
	@$(UV) lock

# =========================
# 新しい実験を作成する
# 使い方: make new NAME=hoge
# =========================
new:
	@if [ -z "$(NAME)" ]; then \
		echo "ERROR: NAMEを指定してください (例: make new NAME=hoge)"; \
		exit 1; \
	fi
	@mkdir -p "$(EXP_ROOT)"
	@LAST=$$(ls -d "$(EXP_ROOT)"/exp[0-9][0-9][0-9]_* 2>/dev/null \
		| sed -E 's#.*/exp([0-9]{3})_.*#\1#' \
		| sort -n \
		| tail -1); \
	if [ -z "$$LAST" ]; then \
		NEXT="000"; \
	else \
		NEXT=$$(printf "%03d" $$(( 10#$$LAST + 1 ))); \
	fi; \
	NEW_DIR="$(EXP_ROOT)/exp$${NEXT}_$(NAME)"; \
	if [ -d "$$NEW_DIR" ]; then \
		echo "ERROR: $$NEW_DIR は既に存在します"; \
		exit 1; \
	fi; \
	echo "create: $$NEW_DIR"; \
	mkdir -p "$$NEW_DIR/exp"; \
	if [ ! -d "$(TEMPLATE)" ]; then \
		echo "ERROR: TEMPLATE が見つかりません: $(TEMPLATE)"; \
		echo "       TEMPLATE を作成するか、TEMPLATE=... を指定してください"; \
		exit 1; \
	fi; \
	if [ -f "$(TEMPLATE)/run.py" ]; then \
		cp "$(TEMPLATE)/run.py" "$$NEW_DIR/"; \
	else \
		echo "Warning: run.py not found in template"; \
	fi; \
	if [ -f "$(TEMPLATE)/config.yaml" ]; then \
		cp "$(TEMPLATE)/config.yaml" "$$NEW_DIR/"; \
	else \
		echo "Warning: config.yaml not found in template"; \
	fi; \
	if [ -f "$(TEMPLATE)/exp/000.yaml" ]; then \
		cp "$(TEMPLATE)/exp/000.yaml" "$$NEW_DIR/exp/000.yaml"; \
	else \
		echo "Warning: exp/000.yaml not found in template"; \
	fi

# =========================
# 実験を実行する（uv）
# 使い方: make run EXP=13 ARGS="..."
# =========================
run:
	@if [ -z "$(EXP)" ]; then \
		echo "ERROR: EXPを指定してください (例: make run EXP=13)"; \
		exit 1; \
	fi
	@EXP_DIR=$$(ls -d "$(EXP_ROOT)"/exp$$(printf "%03d" $(EXP))_* 2>/dev/null); \
	if [ -z "$$EXP_DIR" ]; then \
		echo "ERROR: exp$$(printf "%03d" $(EXP)) が見つかりません"; \
		exit 1; \
	fi; \
	echo "run: $$EXP_DIR/run.py"; \
	cd "$$EXP_DIR" && \
	PROJECT_ROOT="$(PROJECT_ROOT)" \
	PYTHONPATH="$(PROJECT_ROOT)" \
	$(PY) run.py $(ARGS)


# =========================
# 実験を実行する（exp=xxx を指定、uv）
# 使い方: make runy EXP=1 Y=000
#        make runy EXP=1 Y=path/to/000.yaml
# =========================
runy:
	@if [ -z "$(EXP)" ]; then \
		echo "ERROR: EXPを指定してください (例: make runy EXP=1 Y=000)"; \
		exit 1; \
	fi
	@if [ -z "$(Y)" ]; then \
		echo "ERROR: Yを指定してください (例: make runy EXP=1 Y=000)"; \
		exit 1; \
	fi
	@EXP_DIR=$$(ls -d "$(EXP_ROOT)"/exp$$(printf "%03d" $(EXP))_* 2>/dev/null); \
	if [ -z "$$EXP_DIR" ]; then \
		echo "ERROR: exp$$(printf "%03d" $(EXP)) が見つかりません"; \
		exit 1; \
	fi; \
	YRAW="$(Y)"; \
	if [[ "$$YRAW" =~ ^[0-9]+$$ ]]; then \
		YNAME="$$(printf "%03d" $$YRAW)"; \
	else \
		BASE="$$(basename "$$YRAW")"; \
		YNAME="$${BASE%.yaml}"; \
	fi; \
	echo "run: $$EXP_DIR/run.py exp=$$YNAME"; \
	cd "$$EXP_DIR" && \
	PROJECT_ROOT="$(PROJECT_ROOT)" \
	PYTHONPATH="$(PROJECT_ROOT)" \
	$(PY) run.py exp=$$YNAME $(ARGS)

# =========================
# 推論を実行する（uv）
# 使い方: make infer EXP=1 ARGS=""
# =========================
infer:
	@if [ -z "$(EXP)" ]; then \
		echo "ERROR: EXPを指定してください (例: make infer EXP=1)"; \
		exit 1; \
	fi
	@EXP_DIR=$$(ls -d "$(EXP_ROOT)"/exp$$(printf "%03d" $(EXP))_* 2>/dev/null); \
	if [ -z "$$EXP_DIR" ]; then \
		echo "ERROR: exp$$(printf "%03d" $(EXP)) が見つかりません"; \
		exit 1; \
	fi; \
	echo "infer: $$EXP_DIR/infer.py"; \
	cd "$$EXP_DIR" && \
	PROJECT_ROOT="$(PROJECT_ROOT)" \
	PYTHONPATH="$(PROJECT_ROOT)" \
	$(PY) infer.py $(ARGS)

# =========================
# 実験一覧
# =========================
list:
	@ls -d "$(EXP_ROOT)"/exp[0-9][0-9][0-9]_* 2>/dev/null | xargs -n 1 basename || echo "no experiments"

# =========================
# GPU / CUDA チェック
# =========================
gpu-check:
	@echo "=== GPU check (uv) ==="
	@$(PY) -c "\
import torch; \
print(f'torch version      : {torch.__version__}'); \
print(f'cuda available     : {torch.cuda.is_available()}'); \
\
print(f'cuda version       : {torch.version.cuda}') if torch.cuda.is_available() else None; \
print(f'gpu count          : {torch.cuda.device_count()}') if torch.cuda.is_available() else None; \
\
[print(f'  [{i}] {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] \
if torch.cuda.is_available() else print('GPU is NOT available') \
"

