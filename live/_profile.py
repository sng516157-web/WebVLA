"""Time each component of one live-mode step at 640 render."""
import os
import time

import numpy as np

import paths
from live.session import RENDER
from sim.wrapper import DUMMY_ACTION, LiberoWrapper

w = LiberoWrapper("libero_spatial", render_size=RENDER)
w.reset(task_id=0, seed=0)
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL", "unset"))


def bench(name, fn, n=20):
    fn()  # warmup
    t0 = time.time()
    for _ in range(n):
        fn()
    dt = (time.time() - t0) / n * 1000
    print(f"{name}: {dt:.1f} ms")
    return dt


bench("env.step (physics+obs render x2 cams)", lambda: w.step(DUMMY_ACTION))
bench("render(agentview) from cached raw", lambda: w.render("agentview"))
bench("check_success", lambda: w.check_success())

import io
from PIL import Image
frame = w.render("agentview")
crop = frame[80:560]
bench("jpeg encode 640x480", lambda: Image.fromarray(crop).save(io.BytesIO(), format="JPEG", quality=80))
