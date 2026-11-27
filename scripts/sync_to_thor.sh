#!/usr/bin/env bash
set -euo pipefail

# Sync local egogaze project to Jetson Thor
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_HOST="${THOR_HOST:-thor}"
REMOTE_DIR="${THOR_DIR:-~/egogaze}"

echo "Syncing ${PROJECT_DIR} -> ${REMOTE_HOST}:${REMOTE_DIR} ..."
rsync -avz --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache' \
  --exclude 'data/recordings' \
  --exclude 'data/cache' \
  --exclude 'data/models' \
  "${PROJECT_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Done! To run on Thor:"
echo "  ssh -L 8080:localhost:8080 ${REMOTE_HOST} 'cd ${REMOTE_DIR} && PYTHONPATH=src python3 -m egogaze.cli --mode live'"
