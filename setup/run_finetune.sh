#!/bin/bash
# Launch octo finetuning on libero_spatial inside WSL.
# Usage: bash setup/run_finetune.sh
# Paths: override via WEBVLA_HOME / WEBVLA_OCTO / WEBVLA_VENV (see paths.py)
WEBVLA_HOME=${WEBVLA_HOME:-/root/webvla}
WEBVLA_OCTO=${WEBVLA_OCTO:-$WEBVLA_HOME/octo}
WEBVLA_VENV=${WEBVLA_VENV:-$WEBVLA_HOME/venv}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WEBVLA_ENV_FILE=${WEBVLA_ENV_FILE:-$HOME/webvla-env.sh}
[ -f "$WEBVLA_ENV_FILE" ] && source "$WEBVLA_ENV_FILE" || true
export MUJOCO_GL=egl
# offline (not disabled): metrics recorded locally, no network needed;
# requires running WITHOUT --debug (debug forces mode=disabled)
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}   # needs network only on first ckpt download
cd "$WEBVLA_OCTO"
exec "$WEBVLA_VENV/bin/python" scripts/finetune.py \
  --config="$REPO_DIR/setup/finetune_libero_config.py" \
  --name=libero_spatial_bs16_lr5e5
