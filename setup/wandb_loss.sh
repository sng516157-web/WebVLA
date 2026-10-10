#!/bin/bash
# Print training/loss history from the latest offline wandb run in /root/webvla/octo/wandb
cd /root/webvla/octo
RUN=$(ls -dt wandb/offline-run-* | head -1)
echo "run: $RUN"
/root/webvla/venv/bin/wandb sync --view --verbose "$RUN" 2>/dev/null \
  | grep -oE '"training/loss": [0-9.e+-]+|"_step": [0-9]+' \
  | paste - - | awk 'NR % 5 == 1'
