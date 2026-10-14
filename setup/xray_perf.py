#!/usr/bin/env python
"""X-Ray performance gate.

Alternates act() and act_with_internals(all overlays) across replans of one
running episode (scene drift between resets is irrelevant for timing), and
reports the added latency per replan. Budget: < 50 ms added per replan.

Run:  WEBVLA_VENV=/root/webvla/venv_lerobot bash setup/run.sh setup/xray_perf.py
      (set WEBVLA_LEROBOT_DEVICE=cuda to measure the GPU config)
"""

import time

import numpy as np
import torch

from policy.lerobot_pi05_policy import LerobotPi05Policy
from sim.wrapper import LiberoWrapper

N_REPLANS = 8

env = LiberoWrapper("libero_spatial", render_size=256)
policy = LerobotPi05Policy()
obs = env.reset(0, 0)
policy.reset(obs["instruction"])
print(f"device: {policy.device}")

off_ms, on_ms = [], []
chunk = None
for i in range(N_REPLANS * 2):
    pobs = env.obs_pi05()
    torch.manual_seed(1000)  # same noise each call: comparable compute
    t0 = time.perf_counter()
    if i % 2 == 0:
        chunk = policy.act(pobs)
        off_ms.append((time.perf_counter() - t0) * 1000)
    else:
        chunk, _, _ = policy.act_with_internals(
            pobs, want=("denoise", "attn_prefix"))
        on_ms.append((time.perf_counter() - t0) * 1000)
    for a in chunk[:2]:  # advance the scene a little between replans
        env.step(a)

# drop the first of each (warmup)
off, on = np.array(off_ms[1:]), np.array(on_ms[1:])
delta = on - off
print(f"act()                  mean {off.mean():.0f} ms  p95 "
      f"{np.percentile(off, 95):.0f} ms")
print(f"act_with_internals()   mean {on.mean():.0f} ms  p95 "
      f"{np.percentile(on, 95):.0f} ms")
print(f"added per replan       mean {delta.mean():.0f} ms  max {delta.max():.0f} ms")
verdict = "PASS" if delta.mean() < 50 else "OVER BUDGET"
print(f"budget <50 ms/replan: {verdict}")
