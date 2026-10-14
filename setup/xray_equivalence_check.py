#!/usr/bin/env python
"""X-Ray equivalence gate (regression test, READ-ONLY LAW enforcement).

Runs the same episode segment twice — task 0, seed 0, 3 replans x 5 steps:
  A) policy.act()                        (X-Ray fully OFF)
  B) policy.act_with_internals(all on)   (X-Ray fully ON)
and asserts the EXECUTED actions are identical within float tolerance.
Also sanity-checks the captured internals' shapes/ranges.

Run:  WEBVLA_VENV=/root/webvla/venv_lerobot bash setup/run.sh setup/xray_equivalence_check.py
Exit code 0 = pass. This must pass before any X-Ray viz ships and after any
change to policy/xray_internals.py or policy/lerobot_pi05_policy.py.
"""

import numpy as np

from policy.lerobot_pi05_policy import LerobotPi05Policy
from sim.wrapper import LiberoWrapper

TASK_ID, SEED, N_REPLANS = 0, 0, 3
ATOL = 1e-5


def run_segment(with_internals):
    """Fresh env per segment: LIBERO's procedural layout RNG is seeded once
    per env instance (sim/wrapper.py:117), so re-resetting the SAME instance
    drifts the visible scene even though init states pin the objects."""
    import torch

    env = LiberoWrapper("libero_spatial", render_size=256)
    policy = LerobotPi05Policy()
    obs = env.reset(TASK_ID, SEED)
    instruction = obs["instruction"]
    policy.reset(instruction)
    executed = []
    internals_seen = None
    for replan in range(N_REPLANS):
        # Flow matching starts from torch.normal noise on the GLOBAL RNG
        # (modeling_pi05.py:618 sample_noise). Pin the seed per replan so the
        # off/on segments draw identical noise — otherwise both runs are
        # valid but different samples and nothing is comparable.
        torch.manual_seed(1000 + replan)
        pobs = env.obs_pi05()
        if with_internals:
            chunk, full, internals = policy.act_with_internals(
                pobs, want=("denoise", "attn_prefix"))
            internals_seen = internals
            assert np.asarray(full).shape[1] == 7, "full chunk not 7-D"
        else:
            chunk = policy.act(pobs)
        for a in chunk:
            _, _, done, _ = env.step(a)
            executed.append(np.asarray(a, dtype=np.float64))
            if done:
                break
    env.close()
    return np.stack(executed), internals_seen


def main():
    print(f"task {TASK_ID} seed {SEED}, {N_REPLANS} replans per segment",
          flush=True)

    actions_off, _ = run_segment(with_internals=False)
    print(f"[off] executed {len(actions_off)} steps", flush=True)
    actions_on, internals = run_segment(with_internals=True)
    print(f"[on ] executed {len(actions_on)} steps", flush=True)

    assert len(actions_off) == len(actions_on), (
        f"step counts differ: off={len(actions_off)} on={len(actions_on)}")
    diff = np.abs(actions_off - actions_on).max()
    print(f"max |Δaction| between X-Ray off and on: {diff:.3e} "
          f"(tolerance {ATOL:.0e})")
    assert diff < ATOL, (
        "EQUIVALENCE FAILED: X-Ray observation changed the policy's actions. "
        "Do not ship any viz until this passes.")

    assert internals is not None, "no internals captured"
    xt = internals.get("denoise_xt")
    assert xt is not None and xt.ndim == 3 and xt.shape[2] == 7, \
        f"denoise_xt shape wrong: {None if xt is None else xt.shape}"
    print(f"denoise_xt: {xt.shape} (per-step trajectory estimates)")
    for key in ("heatmap_agentview", "heatmap_wrist"):
        h = internals.get(key)
        assert h is not None and h.shape == (256,), f"{key} shape wrong"
        assert 0.0 <= h.min() and h.max() <= 1.0, f"{key} not in [0,1]"
        print(f"{key}: (256,) min={h.min():.3f} max={h.max():.3f} "
              f"argmax patch={int(h.argmax())} (row {int(h.argmax())//16}, "
              f"col {int(h.argmax())%16})")
    print("\nEQUIVALENCE PASS — X-Ray observation does not change actions")


if __name__ == "__main__":
    main()
