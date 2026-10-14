#!/usr/bin/env python
"""Grounding-heatmap visual sanity gate (viz 4 pre-ship check).

Runs one replan with attn_prefix capture on a known task/seed, then renders:
  assets/xray/heatmap_sanity.png  — policy-view frame + 16x16 heatmap overlay
                                    + markers at the referred objects' pixels
and prints the distance from the hottest patch to each referred object.

Coordinate chain (verified against live/session.py CameraGeometry docs):
  mujoco project p -> raw pixel (u right, v up, rows bottom-up) at 256px
  policy image = rotate180(raw): (row, col)_pol = (v, 255 - u), scaled 224/256
  heatmap grid (r, c) center in policy px: ((c+.5)*14, (r+.5)*14)

Run:  WEBVLA_VENV=/root/webvla/venv_lerobot bash setup/run.sh setup/xray_heatmap_sanity.py
Then inspect assets/xray/heatmap_sanity.png — the hot region must sit on or
near the marked referred objects.
"""

import os

import numpy as np

import paths
from policy.lerobot_pi05_policy import LerobotPi05Policy
from sim.wrapper import LiberoWrapper

TASK_ID, SEED = 0, 0
OUT = os.path.join(paths.REPO_ROOT, "assets", "xray", "heatmap_sanity.png")


def project_policy_px(env, point, size=256):
    """world point -> (x, y) in the 224x224 policy image (rotate180 of raw).

    Verified against live/session.py CameraGeometry (selftest-validated):
    raw array row rr = v (rows bottom-up), rotated row r = size-1-v,
    rotated col c = size-1-u. The earlier version of this function had x/y
    swapped and failed exactly this sanity check.
    """
    sim = env.env.sim
    cam_id = sim.model.camera_name2id("agentview")
    pos = np.asarray(sim.data.cam_xpos[cam_id])
    mat = np.asarray(sim.data.cam_xmat[cam_id]).reshape(3, 3)
    fovy = np.deg2rad(float(sim.model.cam_fovy[cam_id]))
    f = (size / 2.0) / np.tan(fovy / 2.0)
    pc = mat.T @ (np.asarray(point, dtype=np.float64) - pos)
    u = f * pc[0] / (-pc[2]) + size / 2.0   # camera x right
    v = f * pc[1] / (-pc[2]) + size / 2.0   # camera y up (raw rows bottom-up)
    col_pol = (size - 1.0) - u
    row_pol = (size - 1.0) - v
    return col_pol * (224.0 / size), row_pol * (224.0 / size)


def main():
    env = LiberoWrapper("libero_spatial", render_size=256)
    policy = LerobotPi05Policy()
    obs = env.reset(TASK_ID, SEED)
    print(f"instruction: {obs['instruction']!r}")
    policy.reset(obs["instruction"])

    _, _, internals = policy.act_with_internals(
        env.obs_pi05(), want=("attn_prefix",))
    heat = internals["heatmap_agentview"].reshape(16, 16)  # (row, col)

    # referred objects: all movable objects, labeled; the instruction names
    # some of them — the hot region should land near one of the named ones
    inner = env.env.env
    marks = []
    for o in inner.objects:
        body_id = inner.obj_body_id[o.name]
        p = np.asarray(env.env.sim.data.body_xpos[body_id], dtype=np.float64)
        x, y = project_policy_px(env, p)
        marks.append((o.name, x, y))
        print(f"  object {o.name:28s} world={p.round(3).tolist()} "
              f"policy_px=({x:.0f},{y:.0f})")

    hot_r, hot_c = np.unravel_index(int(np.argmax(heat)), heat.shape)
    hot_x, hot_y = (hot_c + 0.5) * 14, (hot_r + 0.5) * 14
    print(f"hottest patch: (row {hot_r}, col {hot_c}) -> px ({hot_x:.0f},"
          f" {hot_y:.0f})")
    for name, x, y in marks:
        d = float(np.hypot(hot_x - x, hot_y - y))
        print(f"  distance hottest->[{name}]: {d:.0f} px ({d/14:.1f} patches)")

    # render overlay PNG: policy-view frame + heat + markers
    from PIL import Image, ImageDraw

    frame = np.asarray(env.obs_pi05()["observation/image"])  # policy view
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for r in range(16):
        for c in range(16):
            v = float(heat[r, c])
            od.rectangle([c * 14, r * 14, (c + 1) * 14, (r + 1) * 14],
                         fill=(255, int(140 * (1 - v)), 0, int(115 * v)))
    img = Image.alpha_composite(img, overlay)
    d = ImageDraw.Draw(img)
    for name, x, y in marks:
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(83, 155, 245, 255),
                  width=2)
        d.text((x + 7, y - 6), name.split("_")[0], fill=(83, 155, 245, 255))
    d.ellipse([hot_x - 7, hot_y - 7, hot_x + 7, hot_y + 7],
              outline=(255, 60, 60, 255), width=3)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.convert("RGB").save(OUT)
    print(f"\nwrote {OUT} — inspect: hot region must sit near a named object")


if __name__ == "__main__":
    main()
