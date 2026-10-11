"""Render one reference frame per libero_spatial task into assets/scenes/.

  bash setup/run.sh setup/gen_scene_previews.py

These are committed to the repo so the UI (and the Space) can show what each
scene contains without paying sim startup cost. They use the same upright
180°-rotated view the policy sees.
"""

import os

import imageio.v2 as imageio

import paths
from sim.wrapper import LiberoWrapper

OUT = os.path.join(paths.REPO_ROOT, "assets", "scenes")


def main():
    os.makedirs(OUT, exist_ok=True)
    env = LiberoWrapper("libero_spatial")
    for task_id in range(len(env.benchmark.tasks)):
        env.reset(task_id, 0)
        frame = env.render("agentview")
        path = os.path.join(OUT, f"task{task_id}.png")
        imageio.imwrite(path, frame)
        print(f"task{task_id}: {frame.shape} -> {path} | {env.instruction[:60]}")
    env.close()


if __name__ == "__main__":
    main()
