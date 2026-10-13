#!/bin/bash
# Build /root/webvla/venv_lerobot: LIBERO sim stack (gym 0.25.2, numpy<2) +
# LeRobot 0.4.4 PyTorch pi0.5 inference stack, coexisting in one venv.
# Idempotent; safe to re-run. Requires: ~/webvla-env.sh (proxy), uv at
# /root/.local/bin/uv, LIBERO checkout at /root/webvla/LIBERO,
# openpi-client sources at /root/webvla/openpi (for resize_with_pad in
# sim/wrapper.py:obs_pi05), and the PaliGemma sentencepiece vocab at
# /root/.cache/openpi/big_vision/paligemma_tokenizer.model.
set -euo pipefail
source ~/webvla-env.sh

VENV=/root/webvla/venv_lerobot
UV=/root/.local/bin/uv
PY=$VENV/bin/python

[ -d "$VENV" ] || $UV venv "$VENV" --python 3.10

# --- sim stack (pins must hold: mujoco 2.3.2 / robosuite 1.4.0 / gym 0.25.2, numpy<2)
# termcolor/future/requests/pynput/psutil/matplotlib: undeclared robosuite+bddl+LIBERO deps
$UV pip install --python $PY \
  numpy==1.26.4 mujoco==2.3.2 robosuite==1.4.0 gym==0.25.2 \
  bddl==1.0.1 easydict==1.9 imageio==2.38.0 imageio-ffmpeg==0.6.0 \
  opencv-python==4.6.0.66 cloudpickle==2.1.0 h5py==3.16.0 glfw==2.10.2 \
  scipy==1.11.4 matplotlib==3.7.5 \
  termcolor requests pynput psutil future

# tensorflow-cpu: sim/wrapper.py resize_image() (Octo obs path) imports it lazily
$UV pip install --python $PY tensorflow-cpu==2.15.0

# --- torch 2.8 cu128 (ZeroGPU floor; CPU inference works with these wheels too)
$UV pip install --python $PY torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128

# --- lerobot 0.4.4, --no-deps (full dep list would force numpy>=2 via
# opencv-python-headless>=4.9 and newer huggingface-hub); hand-picked runtime deps:
$UV pip install --python $PY --no-deps lerobot==0.4.4
$UV pip install --python $PY \
  "huggingface-hub==0.35.3" einops "draccus==0.10.0" safetensors deepdiff \
  jsonlines packaging sentencepiece protobuf "gymnasium>=1.1.1,<2.0.0" \
  "datasets>=4.0.0,<5.0.0" "accelerate<2.0.0,>=1.10.0" "diffusers<0.36.0,>=0.27.2" \
  pyserial av

# pi05 needs the lerobot-patched transformers fork (adds
# transformers.models.siglip.check + gemma/paligemma changes the port relies on)
$UV pip install --python $PY \
  "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"

# --- LIBERO + openpi-client source checkouts on sys.path (same mechanism as venv)
SITE=$VENV/lib/python3.10/site-packages
echo "/root/webvla/LIBERO" > "$SITE/libero-repo.pth"
echo "/root/webvla/openpi/packages/openpi-client/src" > "$SITE/openpi-client-repo.pth"

# --- checkpoint (~7GB)
$PY -m huggingface_hub.commands.huggingface_cli download lerobot/pi05_libero_finetuned_v044

echo "venv_lerobot build complete. Smoke test: bash setup/test_lerobot_smoke.sh"
