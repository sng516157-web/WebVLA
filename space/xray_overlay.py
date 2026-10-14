"""X-Ray overlay compositing for the Space (server-side, 256x256 frames).

Live mode (live/xray.js) draws overlays in the browser over a raw socket —
Gradio owns its socket, so on the Space the overlays are drawn INTO the
frames: the live "latest frame" preview and the recorded mp4 get identical
pixels. The default (all-overlays-off) path never calls anything here.

Camera mapping matches live/session.py CameraGeometry exactly (validated by
live/_selftest.py): u = f*pc_x/(-pc_z)+R/2, v = f*pc_y/(-pc_z)+R/2, then the
upright (rotate180) frame has col = R-1-u, row = R-1-v.
"""

import numpy as np
from PIL import Image, ImageDraw

RENDER = 256  # space env render_size
CELL = RENDER // 16  # heatmap grid cell in px


def make_projector(env):
    """world (x,y,z) points -> [(x, y)] pixels in the upright 256x256 frame.
    The closure re-reads camera extrinsics at every call (static camera, but
    nothing is hardcoded)."""
    sim = env.env.sim
    cam_id = sim.model.camera_name2id("agentview")
    fovy = np.deg2rad(float(sim.model.cam_fovy[cam_id]))
    f = (RENDER / 2.0) / np.tan(fovy / 2.0)

    def project(pts):
        pos = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float64)
        mat = np.asarray(sim.data.cam_xmat[cam_id],
                         dtype=np.float64).reshape(3, 3)
        out = []
        for p in pts:
            pc = mat.T @ (np.asarray(p, dtype=np.float64) - pos)
            if pc[2] >= -1e-9:
                continue  # behind the camera
            u = f * pc[0] / (-pc[2]) + RENDER / 2.0
            v = f * pc[1] / (-pc[2]) + RENDER / 2.0
            out.append((float(RENDER - 1.0 - u), float(RENDER - 1.0 - v)))
        return out

    return project


def chunk_waypoints_world(full_chunk, eef_pos):
    """Integrate position deltas (world frame — verified by
    setup/xray_waypoint_check.py, mean direction cos 0.962)."""
    deltas = np.asarray(full_chunk, dtype=np.float64)[:, :3]
    return np.asarray(eef_pos, dtype=np.float64)[None, :] + np.cumsum(
        deltas, axis=0)


def downsample(pts, n=25):
    if len(pts) <= n:
        return pts
    idx = np.linspace(0, len(pts) - 1, n).round().astype(int)
    return [pts[i] for i in idx]


_BAR_LABELS = ["dx", "dy", "dz", "dax", "day", "daz", "grip"]
_BAR_SCALE = [0.05, 0.05, 0.05, 0.35, 0.35, 0.35, 1.0]


def composite(frame_rgb, *, waypoints=None, progress=0.0, action=None,
              denoise=None, heatmap=None, ticks=None, cur_replan=-1):
    """Draw the enabled overlay layers onto one frame. Returns uint8 HWC."""
    img = Image.fromarray(frame_rgb).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    if heatmap is not None:
        grid = np.asarray(heatmap, dtype=np.float32).reshape(16, 16)
        for r in range(16):
            for c in range(16):
                v = float(grid[r, c])
                if v <= 0.02:
                    continue
                d.rectangle([c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL],
                            fill=(255, int(140 * (1 - v)), 0, int(110 * v)))

    if denoise:
        n = len(denoise)
        for i, pts in enumerate(denoise):
            if len(pts) < 2:
                continue
            a = int(30 + 200 * (i / max(n - 1, 1)))
            d.line(pts, fill=(220, 80, 200, a), width=3 if i == n - 1 else 1)

    if waypoints and len(waypoints) >= 2:
        n = len(waypoints)
        for i in range(1, n):
            col = (120, 120, 120, 128) if i / n < progress else (83, 155, 245, 230)
            d.line([waypoints[i - 1], waypoints[i]], fill=col, width=3)
        for i, p in enumerate(waypoints):
            r = 5 if i == n - 1 else 2
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r],
                      fill=(224, 93, 56, 255) if i == n - 1
                      else (83, 155, 245, 230))

    if action is not None:
        bw, bh, x0, y0 = 24, 60, RENDER - (24 + 6) * 7 - 6, 6
        d.rectangle([x0 - 4, y0 - 2, RENDER - 6, y0 + bh + 18],
                    fill=(24, 26, 27, 184))
        for i, v in enumerate(action):
            x = x0 + i * (bw + 6)
            mid = y0 + bh // 2
            h = int(max(-1, min(1, v / _BAR_SCALE[i])) * (bh // 2))
            d.rectangle([x, mid - max(h, 0) if h >= 0 else mid,
                         x + bw, mid if h >= 0 else mid - h],
                        fill=(87, 171, 90, 255) if v >= 0 else (229, 83, 75, 255))
            d.text((x + 4, y0 + bh + 4), _BAR_LABELS[i],
                   fill=(156, 154, 151, 255))

    if ticks:
        n = len(ticks)
        tw = min(220, n * 5)
        x0, y0 = (RENDER - tw) // 2, RENDER - 14
        d.rectangle([x0 - 4, y0 - 8, x0 + tw + 4, y0 + 8],
                    fill=(24, 26, 27, 184))
        mx = max(max(ticks), 1.0)
        for i, ms in enumerate(ticks):
            x = x0 + i * (tw / n)
            h = 2 + 6 * (ms / mx)
            d.rectangle([x, y0 + 6 - h, x + max(2, tw / n - 1), y0 + 6],
                        fill=(224, 93, 56, 255) if i == cur_replan
                        else (83, 155, 245, 255))

    return np.asarray(Image.alpha_composite(img, ov).convert("RGB"))
