#!/usr/bin/env bash
# Quick test: run a single query. Auto-starts llama-server and proxy.
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
"$PROJ_DIR/scripts/setup.sh"

PROJECT="${1:-tqdm}"
QUERY="${2:-How does tqdm detect if running in a Jupyter notebook?}"

source "$PROJ_DIR/scripts/start_services.sh"

echo "Project: $PROJECT"
echo "Query: $QUERY"
echo "---"

cd "$PROJ_DIR"
echo "Analyze the $PROJECT codebase at data/repos/$PROJECT to answer: $QUERY Save your answer to data/outputs/${PROJECT}/test.md" \
  | claude -p \
    --model sonnet \
    --add-dir "./data/repos/$PROJECT" \
    --dangerously-skip-permissions

echo "---"
if [ -f "data/outputs/${PROJECT}/test.md" ]; then
  echo "Output saved to data/outputs/${PROJECT}/test.md"
else
  echo "No output file created (model may have printed to stdout instead)"
fi
