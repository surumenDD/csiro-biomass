FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# uv を入れる
RUN apt-get update && apt-get install -y curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH=/root/.local/bin:$PATH
WORKDIR /work

# 依存関係を先にコピー（キャッシュを効かせる）
COPY requirements.txt .

# システム Python にインストール
RUN uv pip install --system -r requirements.txt

CMD ["bash"]
