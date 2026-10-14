#!/usr/bin/env bash
V=/root/webvla/venv_lerobot
"$V/bin/python" -m py_compile \
  /mnt/c/Users/sng51/Desktop/WebVLA/live/protocol.py \
  /mnt/c/Users/sng51/Desktop/WebVLA/live/app.py \
  /mnt/c/Users/sng51/Desktop/WebVLA/live/session.py \
  /mnt/c/Users/sng51/Desktop/WebVLA/live/xray/engine.py \
  /mnt/c/Users/sng51/Desktop/WebVLA/live/xray/projection.py \
  /mnt/c/Users/sng51/Desktop/WebVLA/live/xray/bundle.py \
  /mnt/c/Users/sng51/Desktop/WebVLA/policy/xray_internals.py \
  /mnt/c/Users/sng51/Desktop/WebVLA/policy/lerobot_pi05_policy.py \
  && echo COMPILE_OK
