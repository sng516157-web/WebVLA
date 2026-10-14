#!/usr/bin/env python
"""Verify the action-delta FRAME CONVENTION used by live/xray/projection.py.

Runs one chunk: predicts cumulative EE waypoints from the chunk's position
deltas, executes the 5 exec actions, and compares predicted vs ACTUAL EE
positions under two hypotheses:
  (a) deltas are in world frame
  (b) deltas are in robot base frame (rotated by base_rot)
Prints per-step errors and asserts the better hypothesis is accurate
(< 1 cm/step). Update live/xray/projection.py FRAME_CONVENTION to match.

Run:  WEBVLA_VENV=/root/webvla/venv_lerobot bash setup/run.sh setup/xray_waypoint_check.py
"""

import numpy as np

from policy.lerobot_pi05_policy import LerobotPi05Policy
from sim.wrapper import LiberoWrapper

TASK_ID, SEED = 0, 0


def base_rotation(env):
    """robot base rotation (world_from_base), if identifiable."""
    sim = env.env.sim
    for name in ("robot0_base", "base", "robot0_base_link"):
        try:
            bid = sim.model.body_name2id(name)
        except ValueError:
            continue
        return np.asarray(sim.data.body_xmat[bid]).reshape(3, 3), name
    return None, None


def main():
    env = LiberoWrapper("libero_spatial", render_size=256)
    policy = LerobotPi05Policy()
    obs = env.reset(TASK_ID, SEED)
    policy.reset(obs["instruction"])

    _, full, _ = policy.act_with_internals(env.obs_pi05(), want=())
    eef0 = np.asarray(env._raw["robot0_eef_pos"], dtype=np.float64)
    rot, rot_name = base_rotation(env)
    print(f"base rotation source: {rot_name}")
    if rot is not None:
        print(f"base rot =\n{np.round(rot, 3)}")

    actual = []
    for a in full[:policy.replan_steps]:
        env.step(a)
        actual.append(np.asarray(env._raw["robot0_eef_pos"], dtype=np.float64))
    actual = np.stack(actual)

    deltas = np.asarray(full[:policy.replan_steps], dtype=np.float64)[:, :3]
    for label, d in (("world", deltas),
                     ("base", deltas @ rot.T if rot is not None else None)):
        if d is None:
            continue
        pred = eef0[None, :] + np.cumsum(d, axis=0)
        errs = np.linalg.norm(pred - actual, axis=1)
        print(f"[{label}] per-step |pred-actual| (m): "
              + " ".join(f"{e:.4f}" for e in errs)
              + f"  max={errs.max():.4f}")
        if errs.max() < 0.01:
            print(f"\nVERDICT: FRAME_CONVENTION = {label!r} "
                  f"(max error {errs.max()*1000:.1f} mm)")
            return
    raise SystemExit("NEITHER frame convention is accurate — investigate "
                     "before shipping the waypoint overlay")


if __name__ == "__main__":
    main()
