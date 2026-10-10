#!/bin/bash
set -uo pipefail
WEBVLA_HOME=${WEBVLA_HOME:-/root/webvla}
WEBVLA_ENV_FILE=${WEBVLA_ENV_FILE:-$HOME/webvla-env.sh}
[ -f "$WEBVLA_ENV_FILE" ] && source "$WEBVLA_ENV_FILE" || true
DST=$WEBVLA_HOME/LIBERO/libero/datasets/libero_spatial
mkdir -p "$DST"
BASE=https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_spatial

FILES=$(curl -s https://huggingface.co/api/datasets/yifengzhu-hf/LIBERO-datasets/tree/main/libero_spatial | grep -o '"path":"libero_spatial/[^"]*"' | cut -d'"' -f4 | sed 's|libero_spatial/||')
echo "files to fetch:"
echo "$FILES"

for f in $FILES; do
  out="$DST/$f"
  for attempt in 1 2 3 4 5 6 7 8; do
    echo ">>> $f (attempt $attempt)"
    curl -sL -C - --retry 5 --retry-delay 3 --connect-timeout 30 --max-time 1200 "$BASE/$f" -o "$out" && break
    sleep 5
  done
done
echo "=== downloaded ==="
ls -la "$DST"
echo "DATASET_DONE"
