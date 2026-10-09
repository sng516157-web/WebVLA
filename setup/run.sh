#!/bin/bash
# Canonical launcher for project scripts inside WSL.
# Usage: bash setup/run.sh <script.py> [args...]
source /root/webvla-env.sh
export MUJOCO_GL=${MUJOCO_GL:-egl}   # EGL verified working; use MUJOCO_GL=osmesa to fall back
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-$MUJOCO_GL}
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
cd /mnt/c/Users/sng51/Desktop/WebVLA
exec /root/webvla/venv/bin/python "$@"
