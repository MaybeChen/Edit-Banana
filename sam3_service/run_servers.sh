#!/usr/bin/env bash
# Start one or more SAM3 HTTP service processes.
# Usage: ./run_servers.sh [N]   (default N=1)
# Ports: 8001, 8002, ... (one per process). Set CUDA_VISIBLE_DEVICES per process if needed.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${CONFIG:-$PROJECT_ROOT/config/config.yaml}"
N="${1:-1}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG (copy config/config.yaml.example to config/config.yaml)"
  exit 1
fi

for ((i=0; i<N; i++)); do
  PORT=$((8001 + i))
  echo "Starting SAM3 service on port $PORT ..."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m sam3_service.server \
    --config "$CONFIG" \
    --port "$PORT" \
    --device "${DEVICE:-auto}" \
    --cache-size "${CACHE_SIZE:-2}" &
done
wait
