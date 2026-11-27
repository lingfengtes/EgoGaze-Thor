#!/usr/bin/env bash
set -euo pipefail

# Sync egogaze to Jetson Thor and launch Web Console with local port forwarding (http://localhost:8080)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-live}"
PORT="${PORT:-8080}"
REMOTE_HOST="${THOR_HOST:-thor}"
REMOTE_DIR="${THOR_DIR:-~/egogaze}"

"${SCRIPT_DIR}/sync_to_thor.sh"

echo "Starting EgoGaze-Thor on ${REMOTE_HOST} (mode=${MODE})..."
echo "Wait ~5s for 'Uvicorn running on http://0.0.0.0:${PORT}', then open: http://localhost:${PORT}"
exec ssh -t -o ServerAliveInterval=15 -L "${PORT}:127.0.0.1:${PORT}" "${REMOTE_HOST}" \
  "cd ${REMOTE_DIR} && PYTHONPATH=src python3 -m egogaze.cli --mode ${MODE} --port ${PORT}"
