#!/bin/bash
# Extended pi05 reference eval: 10 trials/task on libero_spatial (100 episodes).
# Policy server must already be running on port 8000.
set -u
cd /root/webvla/openpi
export MUJOCO_GL=egl
export PYTHONPATH=/root/webvla/LIBERO
exec /root/webvla/venv/bin/python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 10 \
  --args.video-out-path /root/webvla/openpi/data/libero_videos_10 \
  2>&1 | tee /mnt/c/Users/sng51/Desktop/WebVLA/results/pi05_reference_eval_10trials.log
