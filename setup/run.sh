#!/bin/bash
# Canonical launcher for project scripts inside WSL.
# Usage: bash setup/run.sh <script.py | -m module> [args...]
# Paths: override via WEBVLA_HOME / WEBVLA_VENV / WEBVLA_ENV_FILE (see paths.py)
WEBVLA_HOME=${WEBVLA_HOME:-/root/webvla}
WEBVLA_VENV=${WEBVLA_VENV:-$WEBVLA_HOME/venv}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WEBVLA_ENV_FILE=${WEBVLA_ENV_FILE:-$HOME/webvla-env.sh}
[ -f "$WEBVLA_ENV_FILE" ] && source "$WEBVLA_ENV_FILE" || true   # proxy exports (optional)
export MUJOCO_GL=${MUJOCO_GL:-egl}   # EGL verified working; use MUJOCO_GL=osmesa to fall back
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-$MUJOCO_GL}
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
# TF is only used for image resizing (CPU ops); without allow_growth it grabs
# ~9GB of GPU at init, which starves the pi05 policy server (needs ~10.4GB)
# and turns 0.3s inferences into 7s ones.
export TF_FORCE_GPU_ALLOW_GROWTH=${TF_FORCE_GPU_ALLOW_GROWTH:-true}
# All HF assets (octo-base, t5-base tokenizer) are cached; don't depend on the proxy
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"
exec "$WEBVLA_VENV/bin/python" "$@"
