#!/usr/bin/env bash
# Compile-check the Space X-Ray edits + smoke-test the overlay compositing
# end to end on one real replan (env + capture + composite -> PNG).
set -e
V=/root/webvla/venv_lerobot
"$V/bin/python" -m py_compile \
  /mnt/c/Users/sng51/Desktop/WebVLA/space/app.py \
  /mnt/c/Users/sng51/Desktop/WebVLA/space/xray_overlay.py
echo COMPILE_OK

cd /mnt/c/Users/sng51/Desktop/WebVLA
WEBVLA_VENV=/root/webvla/venv_lerobot \
WEBVLA_LEROBOT_DEVICE=${WEBVLA_LEROBOT_DEVICE:-cuda} MUJOCO_GL=egl \
  bash setup/run.sh - <<'PYEOF'
import numpy as np
from PIL import Image

from policy.lerobot_pi05_policy import LerobotPi05Policy
from sim.wrapper import LiberoWrapper
from space.xray_overlay import (chunk_waypoints_world, composite, downsample,
                                make_projector)

env = LiberoWrapper("libero_spatial", render_size=256)
policy = LerobotPi05Policy()
obs = env.reset(0, 0)
policy.reset(obs["instruction"])

chunk, full, internals = policy.act_with_internals(env.obs_pi05())
project = make_projector(env)
eef = np.asarray(env._raw["robot0_eef_pos"], dtype=np.float64)
wps = downsample(project(chunk_waypoints_world(full, eef)))
den = [downsample(project(chunk_waypoints_world(xt, eef)))
       for xt in internals["denoise_xt"]]

frame = env.render("agentview")
out = composite(frame, waypoints=wps, progress=0.4, action=chunk[0],
                denoise=den, heatmap=internals["heatmap_agentview"],
                ticks=[400.0, 380.0, 410.0], cur_replan=2)
assert out.shape == frame.shape and out.dtype == np.uint8
assert not np.array_equal(out, frame), "composite changed nothing"
Image.fromarray(out).save("assets/xray/space_overlay_smoke.png")
# clean path purity: no overlays -> no composite call, frame untouched
print("wps head:", [tuple(round(v, 1) for v in p) for p in wps[:3]])
print("SMOKE OK -> assets/xray/space_overlay_smoke.png")
PYEOF
