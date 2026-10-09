#!/bin/bash
cd /mnt/c/Users/sng51/Desktop/WebVLA
MUJOCO_GL=osmesa /root/webvla/venv/bin/python - <<'EOF'
import numpy as np
from sim.wrapper import LiberoWrapper
from runner.run_episode import run_episode_frames, run_and_save

env = LiberoWrapper("libero_spatial")
obs = env.reset(task_id=0, seed=0)
print("instruction:", obs["instruction"])
print("image_primary:", obs["image_primary"].shape, obs["image_primary"].dtype)
print("image_wrist:", obs["image_wrist"].shape, obs["image_wrist"].dtype)
print("proprio:", obs["proprio"].shape, obs["proprio"].dtype)
frame = env.render("agentview")
print("render:", frame.shape, frame.dtype)

class DummyPolicy:
    def reset(self, instruction): pass
    def act(self, obs):
        return np.tile(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32), (4, 1))

result = {}
frames = list(run_episode_frames(env, DummyPolicy(), task_id=0, seed=0,
                                 max_steps=12, exec_horizon=4, result=result))
print("frames:", len(frames), "result:", {k: result[k] for k in ("success", "steps", "n_replans")})

# determinism check: same (task, seed) -> same initial state
env.reset(task_id=0, seed=0)
p1 = env._raw["robot0_eef_pos"].copy()
env.reset(task_id=0, seed=0)
p2 = env._raw["robot0_eef_pos"].copy()
env.reset(task_id=0, seed=1)
p3 = env._raw["robot0_eef_pos"].copy()
print("seed0==seed0:", np.allclose(p1, p2), "| seed0==seed1:", np.allclose(p1, p3))
env.close()
print("WRAPPER_INTEGRATION_OK")
EOF
