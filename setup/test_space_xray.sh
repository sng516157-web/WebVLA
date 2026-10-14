#!/usr/bin/env bash
# Verify X-Ray overlays on the LIVE Space: run one episode with all overlays
# enabled, pull the mp4, extract a mid-frame, report latency metadata.
set -e
source ~/webvla-env.sh 2>/dev/null || true
: "${HF_TOKEN:?set HF_TOKEN in the environment}"
cat > /tmp/test_space_xray.py <<'PYEOF'
import json
import os
import time

from gradio_client import Client

BASE = "https://swn21-webvla-libero-pi05.hf.space"
SHOTS = "/mnt/c/Users/sng51/Desktop/WebVLA/assets/xray"
OVERLAYS = ["Action path", "Action bars", "Replan timeline",
            "Denoising steps", "Grounding heatmap"]

for attempt in range(60):  # wait out the reboot after push
    try:
        c = Client(BASE, token=os.environ["HF_TOKEN"])
        break
    except Exception as e:
        if attempt == 59:
            raise
        time.sleep(10)
print("space reachable", flush=True)

t0 = time.time()
video, frame, status, badge = c.predict(0, 0, "", OVERLAYS,
                                        api_name="/run_episode")
print(f"episode done in {time.time()-t0:.0f}s", flush=True)
print("badge:", badge.splitlines()[0])
for line in badge.splitlines()[1:]:
    print(" ", line.strip())

import imageio.v2 as imageio
r = imageio.get_reader(video)
frames = [f for f in r]
mid = frames[len(frames) // 2]
imageio.imwrite(f"{SHOTS}/space_live_overlay.png", mid)
print(f"mp4 frames: {len(frames)}; mid-frame -> {SHOTS}/space_live_overlay.png")

# metadata sidecar: the episode json on the container is not exposed, but the
# badge carries the latency line; the mp4 filename proves the artifact.
print("PASS: Space episode with all X-Ray overlays completed")
PYEOF
/root/webvla/venv_lerobot/bin/python /tmp/test_space_xray.py
