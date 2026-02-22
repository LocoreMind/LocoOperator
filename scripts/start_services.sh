#!/usr/bin/env bash
# Start llama-server and proxy, export PIDs for caller to manage.
# Source this file: source scripts/start_services.sh
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="$PROJ_DIR/models/LocoOperator-4B-GGUF/LocoOperator-4B.gguf"
if [ ! -f "$MODEL_PATH" ] && [ ! -L "$MODEL_PATH" ]; then
  echo "ERROR: Model not found at $MODEL_PATH"
  exit 1
fi

_loco_cleanup() {
  [ -n "${PROXY_PID:-}" ] && kill "$PROXY_PID" 2>/dev/null
  [ -n "${LLAMA_PID:-}" ] && kill "$LLAMA_PID" 2>/dev/null
  wait 2>/dev/null
}
trap _loco_cleanup EXIT

# Start llama-server if not already running
if ! curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
  echo "Starting llama-server..."
  llama-server \
    --model "$MODEL_PATH" \
    --port 8080 --ctx-size 32768 -ngl 99 \
    > /dev/null 2>&1 &
  LLAMA_PID=$!
  for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then break; fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
      echo "ERROR: llama-server failed to start"
      exit 1
    fi
    sleep 1
  done
else
  echo "llama-server already running on :8080"
fi

# Start proxy if not already running
if ! curl -s http://127.0.0.1:9091/health > /dev/null 2>&1; then
  echo "Starting proxy..."
  cd "$PROJ_DIR"
  uv run python scripts/proxy.py > /dev/null 2>&1 &
  PROXY_PID=$!
  sleep 2
else
  echo "Proxy already running on :9091"
fi
