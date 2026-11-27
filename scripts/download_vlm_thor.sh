#!/usr/bin/env bash
set -euo pipefail

# Download Qwen3-VL-4B-Instruct onto Jetson Thor (/model/Qwen3-VL-4B-Instruct) via hf-mirror
REMOTE_HOST="${THOR_HOST:-thor}"
TARGET_DIR="${VLM_DIR:-/model/Qwen3-VL-4B-Instruct}"

ssh -t "${REMOTE_HOST}" "
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p '${TARGET_DIR}'
python3 -c '
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id=\"Qwen/Qwen3-VL-4B-Instruct\",
    local_dir=\"${TARGET_DIR}\",
    resume_download=True,
)
print(\"Downloaded Qwen3-VL-4B-Instruct to:\", path)
'
"
