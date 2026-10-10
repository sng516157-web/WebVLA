#!/bin/bash
# Smoke-test the WSL stack: imports, JAX CUDA, MuJoCo GL backends, LIBERO env.
WEBVLA_HOME=${WEBVLA_HOME:-/root/webvla}
WEBVLA_ENV_FILE=${WEBVLA_ENV_FILE:-$HOME/webvla-env.sh}
[ -f "$WEBVLA_ENV_FILE" ] && source "$WEBVLA_ENV_FILE" || true
PY=$WEBVLA_HOME/venv/bin/python

echo "=== 1. core imports ==="
$PY - <<'EOF'
import numpy, mujoco, robosuite
print("numpy", numpy.__version__, "| mujoco", mujoco.__version__, "| robosuite", robosuite.__version__)
import tensorflow as tf
print("tf", tf.__version__)
tf.config.set_visible_devices([], "GPU")
import jax
print("jax", jax.__version__, "devices:", jax.devices())
import torch
print("torch", torch.__version__)
import dlimp, flax, optax
print("dlimp/flax/optax ok, flax", flax.__version__)
EOF

echo "=== 2. octo import ==="
$PY - <<'EOF'
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
from octo.model.octo_model import OctoModel
print("octo import ok")
EOF

echo "=== 3. libero import + benchmark ==="
$PY - <<'EOF'
from libero.libero import benchmark, get_libero_path
bd = benchmark.get_benchmark_dict()
print("suites:", sorted(bd.keys()))
suite = bd["libero_spatial"]()
task = suite.get_task(0)
print("task0:", task.language)
init = suite.get_task_init_states(0)
print("init states:", len(init), type(init[0]))
EOF

echo "=== 4. mujoco GL backends ==="
for backend in egl osmesa; do
  echo "--- MUJOCO_GL=$backend ---"
  MUJOCO_GL=$backend $PY - <<'EOF'
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><geom size='0.1'/></body></worldbody></mujoco>")
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 64, 64)
mujoco.mj_forward(m, d)
r.update_scene(d)
img = r.render()
print("render ok", img.shape, img.dtype)
EOF
done

echo "=== 5. LIBERO env offscreen render (MUJOCO_GL=osmesa) ==="
MUJOCO_GL=osmesa $PY - <<'EOF'
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
obs = env.reset()
init = suite.get_task_init_states(0)
obs = env.set_init_state(init[0])
print("obs keys:", sorted(obs.keys()))
print("agentview:", obs["agentview_image"].shape, obs["agentview_image"].dtype)
print("wrist:", obs["robot0_eye_in_hand_image"].shape)
obs, r, done, info = env.step([0,0,0,0,0,0,-1])
print("step ok, done:", done, "success:", env.check_success())
env.close()
print("LIBERO ENV OK")
EOF
echo "ALL_SMOKE_TESTS_DONE"
