#!/usr/bin/env bash
# Batch analyze queries for a project. Auto-starts llama-server and proxy.
# Usage: ./scripts/analyze.sh <project> [batch_size]
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
"$PROJ_DIR/scripts/setup.sh"

PROJECT="${1:?Usage: $0 <project> [batch_size]}"
BATCH_SIZE="${2:-5}"
TEMPLATE="$PROJ_DIR/prompts/analyze_query.txt"
QUERY_FILE="$PROJ_DIR/data/queries/${PROJECT}-queries.txt"
OUTPUT_DIR="$PROJ_DIR/data/outputs/$PROJECT"

if [ ! -f "$QUERY_FILE" ]; then
  echo "ERROR: query file not found: $QUERY_FILE"
  exit 1
fi

source "$PROJ_DIR/scripts/start_services.sh"

mkdir -p "$OUTPUT_DIR"
PROMPT_TPL=$(cat "$TEMPLATE")

IDS=()
QUESTIONS=()
while IFS=$'\t' read -r id query; do
  [ -z "$id" ] && continue
  IDS+=("$id")
  QUESTIONS+=("$query")
done < "$QUERY_FILE"

echo "Project: $PROJECT"
echo "Total queries: ${#QUESTIONS[@]}, batch size: $BATCH_SIZE"

success=0
skipped=0
failed=0

for ((i=0; i<${#QUESTIONS[@]}; i+=BATCH_SIZE)); do
  batch_num=$(( i/BATCH_SIZE + 1 ))
  echo "=== Batch $batch_num (q$((i+1))-q$((i+BATCH_SIZE))) ==="
  pids=()

  for ((j=i; j<i+BATCH_SIZE && j<${#QUESTIONS[@]}; j++)); do
    id="${IDS[$j]}"
    printf -v num '%03d' "$id"
    question="${QUESTIONS[$j]}"
    output_file="${PROJECT}/q${num}.md"

    if [ -f "$PROJ_DIR/data/outputs/$output_file" ]; then
      skipped=$((skipped + 1))
      continue
    fi

    prompt="${PROMPT_TPL//\{project\}/$PROJECT}"
    prompt="${prompt//\{question\}/$question}"
    prompt="${prompt//\{output_file\}/$output_file}"

    (
      cd "$PROJ_DIR"
      echo "$prompt" | claude -p \
        --model sonnet \
        --add-dir "./data/repos/$PROJECT" \
        --dangerously-skip-permissions > /dev/null 2>&1
      if [ -f "data/outputs/$output_file" ]; then
        echo "  ✓ q${num} done"
      else
        echo "  ✗ q${num} failed"
      fi
    ) &
    pids+=($!)
  done

  for pid in "${pids[@]+"${pids[@]}"}"; do
    [ -z "$pid" ] && continue
    if wait "$pid"; then
      success=$((success + 1))
    else
      failed=$((failed + 1))
    fi
  done
  echo "=== Batch $batch_num complete ==="
done

echo ""
echo "Project: $PROJECT"
echo "Success: $success, Skipped: $skipped, Failed: $failed"
echo "Outputs in data/outputs/$PROJECT/"
