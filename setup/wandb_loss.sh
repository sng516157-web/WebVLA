#!/bin/bash
# Print training/loss history from the latest offline wandb run in $WEBVLA_OCTO/wandb
WEBVLA_HOME=${WEBVLA_HOME:-/root/webvla}
WEBVLA_OCTO=${WEBVLA_OCTO:-$WEBVLA_HOME/octo}
WEBVLA_VENV=${WEBVLA_VENV:-$WEBVLA_HOME/venv}
cd "$WEBVLA_OCTO"
RUN=$(ls -dt wandb/offline-run-* | head -1)
echo "run: $RUN"
"$WEBVLA_VENV/bin/wandb" sync --view --verbose "$RUN" 2>/dev/null \
  | grep -oE '"training/loss": [0-9.e+-]+|"_step": [0-9]+' \
  | paste - - | awk 'NR % 5 == 1'
