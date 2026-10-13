"""Render-backend timing: sim.render at stream size under the current MUJOCO_GL."""
import os
import time

from sim.wrapper import DUMMY_ACTION, LiberoWrapper

w = LiberoWrapper("libero_spatial", render_size=640)
w.reset(task_id=0, seed=0)
w.env.sim.render(640, 480, camera_name="agentview")  # warmup
t0 = time.time()
for _ in range(30):
    img = w.env.sim.render(640, 480, camera_name="agentview")
print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')} render 640x480: "
      f"{(time.time() - t0) / 30 * 1000:.1f} ms", flush=True)
t0 = time.time()
for _ in range(10):
    w.step(DUMMY_ACTION)
print(f"full env.step: {(time.time() - t0) / 10 * 1000:.1f} ms", flush=True)
