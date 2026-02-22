#!/usr/bin/env bash
# Setup script: generates .claude/settings.local.json from .env if not present.
# Run this before using claude -p, or let analyze.sh / test_single.sh call it.
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SETTINGS_FILE="$PROJ_DIR/.claude/settings.local.json"

if [ -f "$SETTINGS_FILE" ]; then
  exit 0
fi

# Load OPENROUTER_API_KEY from .env
if [ ! -f "$PROJ_DIR/.env" ]; then
  echo "ERROR: .env not found. Copy .env.example and set your OPENROUTER_API_KEY."
  exit 1
fi

OPENROUTER_API_KEY=""
while IFS='=' read -r key value; do
  key=$(echo "$key" | xargs)
  [ -z "$key" ] && continue
  [[ "$key" == \#* ]] && continue
  if [ "$key" = "OPENROUTER_API_KEY" ]; then
    OPENROUTER_API_KEY=$(echo "$value" | xargs)
  fi
done < "$PROJ_DIR/.env"

if [ -z "$OPENROUTER_API_KEY" ] || [ "$OPENROUTER_API_KEY" = "sk-or-v1-your-key-here" ]; then
  echo "ERROR: Set a valid OPENROUTER_API_KEY in .env first."
  exit 1
fi

mkdir -p "$PROJ_DIR/.claude"
cat > "$SETTINGS_FILE" <<EOF
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:9091",
    "ANTHROPIC_AUTH_TOKEN": "$OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY": "",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen/qwen3-coder-next",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4-5-20251001",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen/qwen3-coder-next"
  }
}
EOF

echo "Generated $SETTINGS_FILE"
