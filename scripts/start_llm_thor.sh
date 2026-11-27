#!/usr/bin/env bash
set -euo pipefail

# Start TensorRT-Edge-LLM C++ daemon (Qwen3-8B-NVFP4) on Jetson Thor port 8000
# Includes automatic restart loop in case of transient disconnection.
REMOTE_HOST="${THOR_HOST:-thor}"
PORT="${LLM_PORT:-8000}"

ssh -t "${REMOTE_HOST}" "
export EDGELLM_PLUGIN_PATH=\$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
while true; do
  echo '[start_llm_thor] Starting Qwen3-8B-NVFP4 daemon on 127.0.0.1:${PORT}...'
  \$HOME/TensorRT-Edge-LLM/build/examples/llm/llm_server_daemon \
    --engineDir=/model/Qwen3-8B-NVFP4/engines \
    --port=${PORT} \
    --host=127.0.0.1 || true
  echo '[start_llm_thor] Daemon exited; restarting in 2s (Ctrl+C to stop)...'
  sleep 2
done
"
