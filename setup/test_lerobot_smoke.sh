#!/bin/bash
# CPU smoke test for policy/lerobot_pi05_policy.py (LeRobot PyTorch pi0.5).
#
# Run from Git Bash on the Windows host:
#   MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/c/Users/sng51/Desktop/WebVLA/setup/test_lerobot_smoke.sh
# or inside WSL directly:
#   bash setup/test_lerobot_smoke.sh
#
# Uses /root/webvla/venv_lerobot (see setup/_lerobot_scratch/ for how it was built).
set -euo pipefail
source ~/webvla-env.sh  # proxy for HF downloads

export MUJOCO_GL=egl
export PYTHONPATH="/mnt/c/Users/sng51/Desktop/WebVLA:${PYTHONPATH:-}"
# LIBERO checkout + openpi-client are on sys.path via .pth files in the venv.
PY=/root/webvla/venv_lerobot/bin/python

$PY - <<'EOF'
import os
import time

import numpy as np

t0 = time.time()
from sim.wrapper import LiberoWrapper

env = LiberoWrapper("libero_spatial")
obs_octo = env.reset(0, 0)  # Octo-format obs (exercises the TF resize path too)
instruction = obs_octo["instruction"]
print(f"[smoke] env ready ({time.time()-t0:.1f}s), instruction: {instruction!r}", flush=True)

t0 = time.time()
from policy.lerobot_pi05_policy import LerobotPi05Policy

policy = LerobotPi05Policy(device="cpu")
print(f"[smoke] policy loaded ({time.time()-t0:.1f}s)", flush=True)

policy.reset(instruction)
obs = env.obs_pi05()

t0 = time.time()
actions = policy.act(obs)
dt = time.time() - t0

print(f"[smoke] act() took {dt:.1f}s (first call, CPU)", flush=True)
print("[smoke] actions shape:", actions.shape, "dtype:", actions.dtype)
print("[smoke] all finite:", np.isfinite(actions).all())
print("[smoke] first action :", np.array2string(actions[0], precision=4))
print("[smoke] second action:", np.array2string(actions[1], precision=4))
assert actions.shape == (5, 7), f"expected (5, 7), got {actions.shape}"
assert actions.dtype == np.float32
assert np.isfinite(actions).all()
print("[smoke] OK")
env.close()
EOF
