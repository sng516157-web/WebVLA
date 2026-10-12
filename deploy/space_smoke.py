"""Build-time smoke test for the Space image (CPU, OSMesa).

Exercises the whole closed-loop stack so a broken image fails at BUILD time:
LIBERO data files (.bddl/.init), robosuite env creation, OSMesa offscreen
rendering, TF image resize, checkpoint load from the HF repo, and one policy
forward pass (jax compile included).

Run at the end of the docker build:  RUN python space_smoke.py
"""

import os

import numpy as np


def main():
    from sim.wrapper import LiberoWrapper

    w = LiberoWrapper()
    obs = w.reset(0, 0)
    frame = w.render("agentview")
    print("ENV OK:", obs["image_primary"].shape, obs["image_wrist"].shape,
          frame.shape, "| instruction:", obs["instruction"])
    w.close()

    from policy.octo_policy import OctoPolicy

    p = OctoPolicy(os.environ["WEBVLA_CKPT"])
    p.reset("pick up the black bowl and place it on the plate")
    a = p.act({k: v for k, v in obs.items() if isinstance(v, np.ndarray)})
    print("POLICY OK: action chunk", a.shape, "| unnorm key:", p.unnorm_key)


if __name__ == "__main__":
    main()
