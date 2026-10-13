#!/usr/bin/env bash
# Launch the ZeroGPU Space app locally (short episode for smoke testing).
set -e
cd /mnt/c/Users/sng51/Desktop/WebVLA
export WEBVLA_VENV=/root/webvla/venv_lerobot
export WEBVLA_LEROBOT_DEVICE=cuda
export WEBVLA_SPACE_MAX_STEPS=10
export PYTHONPATH=.:space
exec bash setup/run.sh space/app.py
