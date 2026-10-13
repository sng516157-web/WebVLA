#!/bin/bash
# Official openpi LIBERO eval: pi05_libero on libero_spatial, 10 tasks x 3 trials.
set -uo pipefail
LOG=/mnt/c/Users/sng51/Desktop/WebVLA/results/pi05_reference_eval.log
mkdir -p /mnt/c/Users/sng51/Desktop/WebVLA/results
cd /root/webvla/openpi
MUJOCO_GL=egl PYTHONPATH=/root/webvla/LIBERO \
  /root/webvla/venv/bin/python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 3 \
  --args.video-out-path /root/webvla/openpi/data/libero_videos \
  2>&1 | tee "$LOG"
