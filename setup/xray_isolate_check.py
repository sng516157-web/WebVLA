#!/usr/bin/env python
"""Isolate the equivalence failure: model capture vs env reset drift.

1. env reset reproducibility: fresh LiberoWrapper per reset (LIBERO's layout
   RNG is seeded once per env instance, so same-instance re-resets drift);
   compare obs images + states bitwise.
2. capture neutrality: on ONE fixed obs, seed torch RNG identically, call
   act() vs act_with_internals(), compare exec chunks.

Run:  WEBVLA_VENV=/root/webvla/venv_lerobot bash setup/run.sh setup/xray_isolate_check.py
"""

import numpy as np
import torch

from policy.lerobot_pi05_policy import LerobotPi05Policy
from sim.wrapper import LiberoWrapper

policy = LerobotPi05Policy()


def fresh_obs():
    env = LiberoWrapper("libero_spatial", render_size=256)
    env.reset(0, 0)
    return env, env.obs_pi05()


# --- 1. reset reproducibility (fresh env per reset) ------------------------
_, a = fresh_obs()
_, b = fresh_obs()
for k in a:
    if k == "instruction":
        continue
    d = np.abs(np.asarray(a[k], dtype=np.float64)
               - np.asarray(b[k], dtype=np.float64)).max()
    print(f"reset reproducibility {k}: max|Δ| = {d:.3e}")

# --- 2. capture neutrality on a fixed obs ---------------------------------
policy.reset(a["instruction"])
torch.manual_seed(1000)
c1 = policy.act(a)
torch.manual_seed(1000)
c2, full, internals = policy.act_with_internals(a)
d = np.abs(c1 - c2).max()
print(f"capture neutrality on fixed obs: max|Δ| = {d:.3e}")
print(f"denoise_xt captured: {internals.get('denoise_xt', np.zeros(0)).shape}")
print(f"heatmap_agentview captured: {'heatmap_agentview' in internals}")
print(f"heatmap_wrist captured: {'heatmap_wrist' in internals}")
