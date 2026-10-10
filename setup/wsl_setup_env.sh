#!/bin/bash
# Bootstrap the WebVLA M0 environment inside WSL2 Ubuntu.
# - Python 3.10 venv at $WEBVLA_HOME/venv (via uv-managed CPython)
# - LIBERO pinned sim stack (robosuite==1.4.0, mujoco==2.3.2) + Octo stack (jax[cuda12]==0.4.20, flax==0.7.5)
#
# Documented deviations from LIBERO requirements.txt (forced by hard conflicts, sim stack untouched):
#   numpy 1.22.4 -> 1.24.3   (tensorflow 2.15 requires >=1.23.5; octo pins ==1.24.3)
#   einops 0.4.1 -> >=0.6.1  (octo requires >=0.6.1; einops not used by LIBERO env code)
#   transformers not installed (LIBERO training-only dep; octo eval path does not import it)
#   robomimic/wandb/torch not installed (LIBERO training-only deps)
set -euo pipefail
WEBVLA_HOME=${WEBVLA_HOME:-/root/webvla}
WEBVLA_ENV_FILE=${WEBVLA_ENV_FILE:-$HOME/webvla-env.sh}
[ -f "$WEBVLA_ENV_FILE" ] && source "$WEBVLA_ENV_FILE" || true
source /root/.local/bin/env

PY=$WEBVLA_HOME/venv/bin/python

uv venv $WEBVLA_HOME/venv --python 3.10

# --- LIBERO pinned sim stack ---
uv pip install --python $PY \
  numpy==1.24.3 \
  mujoco==2.3.2 \
  robosuite==1.4.0 \
  bddl==1.0.1 \
  easydict==1.9 \
  cloudpickle==2.1.0 \
  gym==0.25.2 \
  opencv-python==4.6.0.66 \
  future==0.18.2 \
  hydra-core==1.2.0 \
  imageio imageio-ffmpeg \
  h5py

uv pip install --python $PY --no-deps -e $WEBVLA_HOME/LIBERO

# --- Octo stack ---
uv pip install --python $PY "jax[cuda12_pip]==0.4.20" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

uv pip install --python $PY \
  flax==0.7.5 \
  optax==0.1.5 \
  chex==0.1.85 \
  ml_dtypes==0.2.0 \
  distrax==0.1.5 \
  tensorflow-cpu==2.15.0 \
  tensorflow_probability==0.23.0 \
  einops==0.6.1 \
  absl-py tqdm ml_collections scipy \
  huggingface_hub

# --- dlimp (pinned commit; git protocol is blocked by proxy, use tarball) ---
cd "$WEBVLA_HOME"
if [ ! -d dlimp-src ]; then
  curl -sL https://codeload.github.com/kvablack/dlimp/tar.gz/5edaa4691567873d495633f2708982b42edf1972 -o dlimp.tar.gz
  tar xzf dlimp.tar.gz
  mv dlimp-5edaa4691567873d495633f2708982b42edf1972 dlimp-src
  rm dlimp.tar.gz
fi
uv pip install --python $PY ./dlimp-src

uv pip install --python $PY --no-deps -e $WEBVLA_HOME/octo

echo "SETUP_DONE"
