#!/usr/bin/env bash
# Runs once per cluster from serve.yaml setup. Needs MODEL.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

sudo apt-get update -qq && sudo apt-get install -y -qq python3-dev
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ~/vllm-env ] || uv venv ~/vllm-env --python 3.12
uv pip install --python ~/vllm-env/bin/python "vllm==0.29.0" pandas datasets
~/vllm-env/bin/vllm --version

sudo docker compose version || curl -fsSL https://get.docker.com | sudo sh

~/vllm-env/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', ignore_patterns=['consolidated*', '*.pth', 'original/*', '*.gguf'])
"
