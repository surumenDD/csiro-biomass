# Makefile (docker compose + experiments)

SERVICE := dev
WORKDIR ?= /work

DOCKER_COMPOSE := docker compose
RUN := $(DOCKER_COMPOSE) run --rm --workdir $(WORKDIR) $(SERVICE)

EXP_ROOT := experiments
TEMPLATE := $(EXP_ROOT)/exp000_sample
PYTHON := python

SHELL := /bin/bash

.PHONY: help \
	build up shell gpu-check python logs down clean \
	new run list

help:
	@echo "Targets:"
	@echo "  build      docker compose build"
	@echo "  up         docker compose up -d"
	@echo "  shell      docker compose run --rm で bash"
	@echo "  gpu-check  torch CUDA チェック"
	@echo "  python     python を起動"
	@echo "  logs       ログ表示"
	@echo "  down       停止"
	@echo "  clean      生成物と未使用イメージの掃除"
	@echo ""
	@echo "Experiments:"
	@echo "  new        新しい実験を作成 (例: make new NAME=hoge)"
	@echo "  run        実験を実行 (例: make run EXP=13)"
	@echo "  list       実験一覧を表示"

# =========================
# docker compose
# =========================
build:
	$(DOCKER_COMPOSE) build

up:
	$(DOCKER_COMPOSE) up -d

shell:
	$(RUN) bash

gpu-check:
	$(RUN) $(PYTHON) -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('is_available', torch.cuda.is_available()); print('name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

python:
	$(RUN) $(PYTHON)

logs:
	$(DOCKER_COMPOSE) logs -f --tail=200

down:
	$(DOCKER_COMPOSE) down

clean:
	$(DOCKER_COMPOSE) down --remove-orphans
	-docker image prune -f

# =========================
# experiments（ホスト側でファイル作成）
# =========================
new:
	@if [ -z "$(NAME)" ]; then \
		echo "ERROR: NAMEを指定してください (例: make new NAME=hoge)"; \
		exit 1; \
	fi
	@mkdir -p $(EXP_ROOT)
	@LAST=$$(ls -d $(EXP_ROOT)/exp[0-9][0-9][0-9]_* 2>/dev/null \
		| sed -E 's#.*/exp([0-9]{3})_.*#\1#' \
		| sort -n \
		| tail -1); \
	if [ -z "$$LAST" ]; then \
		NEXT="000"; \
	else \
		NEXT=$$(printf "%03d" $$(( 10#$$LAST + 1 ))); \
	fi; \
	NEW_DIR=$(EXP_ROOT)/exp$${NEXT}_$(NAME); \
	if [ -d "$$NEW_DIR" ]; then \
		echo "ERROR: $$NEW_DIR は既に存在します"; \
		exit 1; \
	fi; \
	echo "create: $$NEW_DIR"; \
	mkdir -p $$NEW_DIR/exp; \
	cp $(TEMPLATE)/run.py $$NEW_DIR/ 2>/dev/null || echo "Warning: run.py not found in template"; \
	cp $(TEMPLATE)/config.yaml $$NEW_DIR/ 2>/dev/null || echo "Warning: config.yaml not found in template"; \
	cp $(TEMPLATE)/exp/000.yaml $$NEW_DIR/exp/000.yaml 2>/dev/null || echo "Warning: 000.yaml not found in template"

# =========================
# experiments（コンテナ内で実行）
# =========================
run:
	@if [ -z "$(EXP)" ]; then \
		echo "ERROR: EXPを指定してください (例: make run EXP=13)"; \
		exit 1; \
	fi
	@EXP_DIR=$$(ls -d $(EXP_ROOT)/exp$$(printf "%03d" $(EXP))_* 2>/dev/null); \
	if [ -z "$$EXP_DIR" ]; then \
		echo "ERROR: exp$$(printf "%03d" $(EXP)) が見つかりません"; \
		exit 1; \
	fi; \
	echo "run: $$EXP_DIR/run.py (in container)"; \
	$(RUN) bash -lc "cd $$EXP_DIR && $(PYTHON) run.py"

# =========================
# experiments（一覧）
# =========================
list:
	@ls -d $(EXP_ROOT)/exp[0-9][0-9][0-9]_* 2>/dev/null | xargs -n 1 basename || echo "no experiments"
