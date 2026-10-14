#!/usr/bin/env bash
V=/root/webvla/venv_lerobot
for mod in fastapi uvicorn websockets mujoco robosuite libero imageio PIL torch lerobot jax; do
  "$V/bin/python" -c "import $mod" 2>/dev/null && echo "$mod: OK" || echo "$mod: MISSING"
done
echo "== octo_policy imports =="
head -30 /mnt/c/Users/sng51/Desktop/WebVLA/policy/octo_policy.py | grep -n "^import\|^from"
echo "== pi0_policy imports =="
head -30 /mnt/c/Users/sng51/Desktop/WebVLA/policy/pi0_policy.py | grep -n "^import\|^from"
