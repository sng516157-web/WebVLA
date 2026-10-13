#!/usr/bin/env bash
# Full-episode smoke against the LIVE ZeroGPU Space via the public Gradio API.
set -e
source ~/webvla-env.sh
cd /mnt/c/Users/sng51/Desktop/WebVLA
cat > /tmp/space_e2e.py <<'EOF'
from gradio_client import Client

c = Client("https://swn21-webvla-libero-pi05.hf.space")
print("submitting full episode (task 0, seed 0) to the LIVE Space ...", flush=True)
job = c.submit(0, 0, "", api_name="/run_episode")
video, frame, status, badge = job.result()
print("status:", status)
print("badge:", badge)
print("video:", video)
assert video, "no video returned"
print("PASS: full episode completed on the ZeroGPU Space")
EOF
exec /root/webvla/venv_lerobot/bin/python /tmp/space_e2e.py
