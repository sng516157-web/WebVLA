#!/usr/bin/env bash
# Smoke: LerobotPi05Policy on hub 2.x + gradio 6.30 (Space dep combo).
set -e
cd /mnt/c/Users/sng51/Desktop/WebVLA
export WEBVLA_VENV=/root/webvla/venv_lerobot
export WEBVLA_LEROBOT_DEVICE=cuda
cat > /tmp/hub2_smoke.py <<'EOF'
import numpy as np
from policy.lerobot_pi05_policy import LerobotPi05Policy

p = LerobotPi05Policy(device="cuda")
p.reset("pick up the black bowl and place it on the plate")
obs = {
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(8, dtype=np.float32),
}
chunk = p.act(obs)
print("chunk:", chunk.shape, chunk.dtype)
assert chunk.shape == (5, 7)
print("PASS: fork works on huggingface_hub 2.x")
EOF
exec bash setup/run.sh /tmp/hub2_smoke.py
