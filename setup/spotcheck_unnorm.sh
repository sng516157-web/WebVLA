#!/usr/bin/env bash
set -e
V=/root/webvla/venv_lerobot/lib/python3.10/site-packages
echo "== UnnormalizerProcessorStep =="
sed -n '430,540p' "$V/lerobot/processor/normalize_processor.py"
echo "== pipeline __call__ (tensor passthrough?) =="
grep -n "def __call__" "$V/lerobot/processor/pipeline.py" | head -3
