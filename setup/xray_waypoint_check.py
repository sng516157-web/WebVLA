#!/usr/bin/env python
"""Frame-convention check v2: compare per-step DIRECTION (cosine) and
magnitude ratio between commanded position deltas and achieved EE motion.

The OSC impedance controller tracks commanded deltas asymptotically, so
magnitudes lag; the frame convention is validated by direction alignment.

Run:  WEBVLA_VENV=/root/webvla/venv_lerobot bash setup/run.sh setup/xray_waypoint_check.py
"""

import numpy as np

from policy.lerobot_pi05_policy import LerobotPi05Policy
from sim.wrapper import LiberoWrapper

TASK_ID, SEED = 0, 0


def main():
    env = LiberoWrapper("libero_spatial", render_size=256)
    policy = LerobotPi05Policy()
    obs = env.reset(TASK_ID, SEED)
    policy.reset(obs["instruction"])

    _, full, _ = policy.act_with_internals(env.obs_pi05(), want=())
    eef = np.asarray(env._raw["robot0_eef_pos"], dtype=np.float64)

    print(f"{'k':>2} {'|cmd|mm':>8} {'|ach|mm':>8} {'cos':>6}")
    cos_ok, n_dir = [], 0
    for k, a in enumerate(full[:policy.replan_steps]):
        env.step(a)
        new = np.asarray(env._raw["robot0_eef_pos"], dtype=np.float64)
        cmd = np.asarray(a[:3], dtype=np.float64)
        ach = new - eef
        eef = new
        nc, na = np.linalg.norm(cmd), np.linalg.norm(ach)
        cos = float(cmd @ ach / (nc * na)) if nc > 1e-9 and na > 1e-9 else float("nan")
        print(f"{k:>2} {nc*1000:8.1f} {na*1000:8.1f} {cos:6.3f}")
        if nc > 1e-4 and na > 1e-4:  # only where direction is meaningful
            cos_ok.append(cos)
            n_dir += 1
    if n_dir and float(np.mean(cos_ok)) > 0.9:
        print(f"\nVERDICT: world-frame deltas, direction-aligned "
              f"(mean cos {np.mean(cos_ok):.3f} over {n_dir} steps). "
              "Tracking lag is the controller's, not a frame bug.")
        return
    raise SystemExit(f"direction mismatch (mean cos "
                     f"{np.mean(cos_ok) if cos_ok else float('nan'):.3f}) — "
                     "do not ship the waypoint overlay")


if __name__ == "__main__":
    main()
