#!/bin/bash
cd /mnt/c/Users/sng51/Desktop/WebVLA
export MUJOCO_GL=egl
/root/webvla/venv/bin/python - <<'EOF'
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from sim.wrapper import LiberoWrapper
from policy.octo_policy import OctoPolicy

env = LiberoWrapper("libero_spatial")
policy = OctoPolicy("hf://rail-berkeley/octo-base-1.5")
print("unnorm_key:", policy.unnorm_key)
print("expected obs:", policy.expected_observation)

obs = env.reset(task_id=0, seed=0)
policy.reset(obs["instruction"])
import imageio.v2 as imageio
imageio.imwrite("setup/frame0.png", env.render("agentview"))
actions = policy.act(obs)
print("action chunk shape:", actions.shape)
print("chunk:\n", np.round(actions, 3))
for a in actions:
    env.step(a)
imageio.imwrite("setup/frame1.png", env.render("agentview"))
actions2 = policy.act(env.obs())
print("second chunk:\n", np.round(actions2, 3))
env.close()
print("POLICY_LOOP_OK")
EOF
