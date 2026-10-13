#!/usr/bin/env bash
# One-shot pi05 policy server health check: connect, run N dummy inferences
# with the exact LIBERO obs shape, report latencies. Exits non-zero on failure.
set -euo pipefail
cd /mnt/c/Users/sng51/Desktop/WebVLA
exec bash setup/run.sh - <<'PY'
import time
import numpy as np
from openpi_client import websocket_client_policy

t0 = time.time()
client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000)
print(f"connect: {time.time()-t0:.1f}s", flush=True)

element = {
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(8, dtype=np.float32),
    "prompt": "pick up the black bowl and place it on the plate",
}
for i in range(3):
    t0 = time.time()
    out = client.infer(element)
    dt = time.time() - t0
    a = np.asarray(out["actions"])
    print(f"infer {i}: {dt:.2f}s actions.shape={a.shape} finite={np.isfinite(a).all()}", flush=True)
print("OK")
PY
