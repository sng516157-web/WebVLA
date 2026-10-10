#!/bin/bash
# Launch octo finetuning on libero_spatial inside WSL.
# Usage: bash setup/run_finetune.sh
source /root/webvla-env.sh
export MUJOCO_GL=egl
# offline (not disabled): metrics recorded locally, no network needed;
# requires running WITHOUT --debug (debug forces mode=disabled)
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
cd /root/webvla/octo
exec /root/webvla/venv/bin/python scripts/finetune.py \
  --config=/mnt/c/Users/sng51/Desktop/WebVLA/setup/finetune_libero_config.py \
  --name=libero_spatial_bs16_lr5e5
