"""Side-by-side panel rendering for Space X-Ray (matplotlib Agg -> numpy).

Panels mode produces a second video where the sim frame is left untouched
and the plots sit beside it in a grid — so replaying the episode replays the
panels in sync. Uses fig.canvas.buffer_rgba (not savefig) with fixed axes
rects: ~5 ms per panel refresh, cheap enough to run per frame.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

_BAR_LABELS = ["dx", "dy", "dz", "dax", "day", "daz", "grip"]

# combo grid: 2 columns x 3 rows of cells; sim video always in cell 0
CELL_W, CELL_H = 288, 216
CELL_ORDER = ["sim", "path", "denoise", "bars", "timeline", "heatmap"]
CELL_TITLES = {
    "sim": "simulation",
    "path": "plan view (3D)",
    "denoise": "denoising convergence",
    "bars": "action bars",
    "timeline": "replan timeline",
    "heatmap": "attention heatmap",
}


class PanelRenderer:
    def __init__(self):
        self._figs = {}

    def _fig(self, key):
        if key not in self._figs:
            self._figs[key] = plt.figure(figsize=(2.88, 2.16), dpi=100)
        fig = self._figs[key]
        fig.clear()
        return fig

    @staticmethod
    def _to_rgb(fig):
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    @staticmethod
    def _ax(fig, title):
        ax = fig.add_axes([0.14, 0.16, 0.80, 0.70])
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        return ax

    def close(self):
        for fig in self._figs.values():
            plt.close(fig)
        self._figs.clear()

    # ------------------------------------------------------------- panels
    def bars(self, action):
        fig = self._fig("bars")
        ax = self._ax(fig, "action being executed (7-D)")
        a = np.asarray(action, dtype=float)
        colors = ["#57ab5a" if v >= 0 else "#e5534b" for v in a]
        ax.bar(range(7), a, color=colors)
        ax.set_xticks(range(7), _BAR_LABELS, fontsize=7)
        ax.axhline(0, color="#373a3d", lw=0.8)
        lim = max(float(np.max(np.abs(a))), 0.05)
        ax.set_ylim(-1.2 * lim, 1.2 * lim)
        return self._to_rgb(fig)

    def timeline(self, ticks, cur):
        fig = self._fig("timeline")
        ax = self._ax(fig, "replan timeline (orange = executing)")
        n = len(ticks)
        show = list(range(max(0, n - 40), n))
        colors = ["#e05d38" if i == cur else "#539bf5" for i in show]
        ax.bar(show, [ticks[i] for i in show], color=colors)
        ax.set_xlabel("replan tick", fontsize=7)
        ax.set_ylabel("inference ms", fontsize=7)
        return self._to_rgb(fig)

    @staticmethod
    def _trusted_limits(ax, anchors):
        """Frame the 3D axes on the physically-realized region (executed
        trail, gripper) with padding — never on the full imagined chunk,
        whose unvisited tail drifts meters away under distribution shift
        and would squash the readable region to a dot. Imagined points
        outside the box clip at the axes — the honest equivalent of the
        video overlay leaving the frame."""
        pts = np.asarray([p for a in anchors if a is not None
                          for p in np.atleast_2d(a)], dtype=float)
        if len(pts) == 0:
            return
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = np.maximum(hi - lo, 0.25)
        pad = 0.05
        ax.set_xlim(lo[0] - pad, lo[0] + span[0] + pad)
        ax.set_ylim(lo[1] - pad, lo[1] + span[1] + pad)
        ax.set_zlim(lo[2] - pad, lo[2] + span[2] + pad)
        ax.set_box_aspect((span[0] + 2 * pad, span[1] + 2 * pad,
                           span[2] + 2 * pad))

    def path(self, plan_xyz, trail_xyz, eef_xyz):
        """3D view: the imagined 50-step plan vs the path actually executed
        this episode, plus the gripper now. Fixed perspective."""
        fig = self._fig("path")
        ax = fig.add_axes([0.0, 0.0, 1.0, 0.88], projection="3d")
        ax.set_title("plan view (3D, world m)", fontsize=8)
        if trail_xyz is not None and len(trail_xyz) > 1:
            t = np.asarray(trail_xyz)
            ax.plot(t[:, 0], t[:, 1], t[:, 2], "-", color="#57ab5a", lw=2,
                    label="executed")
        if plan_xyz is not None and len(plan_xyz) > 1:
            p = np.asarray(plan_xyz)
            ax.plot(p[:, 0], p[:, 1], p[:, 2], ".-", color="#539bf5",
                    lw=1.5, ms=3, label="imagined plan (50)")
        if eef_xyz is not None:
            ax.scatter(*eef_xyz, color="#e05d38", s=40, label="gripper")
        ax.legend(fontsize=6, loc="upper left")
        self._trusted_limits(ax, [trail_xyz, eef_xyz,
                                  plan_xyz[:2] if plan_xyz is not None
                                  else None])
        ax.view_init(elev=25, azim=-60)
        ax.tick_params(labelsize=5)
        return self._to_rgb(fig)

    def denoise(self, trajs_xyz):
        """3D view: the 10 flow-matching estimates of the same plan,
        faint -> solid."""
        fig = self._fig("denoise")
        ax = fig.add_axes([0.0, 0.0, 1.0, 0.88], projection="3d")
        ax.set_title("denoising: 10 refinements of the plan (3D)", fontsize=8)
        n = len(trajs_xyz)
        for i, xyz in enumerate(trajs_xyz):
            if len(xyz) < 2:
                continue
            t = np.asarray(xyz)
            alpha = 0.15 + 0.85 * (i / max(n - 1, 1))
            ax.plot(t[:, 0], t[:, 1], t[:, 2], "-",
                    color=(0.86, 0.31, 0.78, alpha),
                    lw=2 if i == n - 1 else 1)
        anchor = trajs_xyz[-1][:2] if n and len(trajs_xyz[-1]) else None
        self._trusted_limits(ax, [anchor])
        ax.view_init(elev=25, azim=-60)
        ax.tick_params(labelsize=5)
        return self._to_rgb(fig)

    def heatmap(self, grid):
        fig = self._fig("heatmap")
        ax = fig.add_axes([0.06, 0.06, 0.88, 0.76])
        ax.set_title("instruction attention over image patches", fontsize=8)
        ax.imshow(np.asarray(grid, dtype=float).reshape(16, 16),
                  cmap="hot", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        return self._to_rgb(fig)


def compose_combo(frame_rgb, cells):
    """frame_rgb (256x256 sim) + {cell_name: panel_rgb|None} -> one wide
    frame (576x648): sim top-left, panels in a fixed grid; disabled panels
    get a dark cell so the layout never jumps around."""
    canvas = Image.new("RGB", (CELL_W * 2, CELL_H * 3), (24, 26, 27))
    draw = ImageDraw.Draw(canvas)
    cell_imgs = {"sim": frame_rgb}
    cell_imgs.update(cells)
    for i, name in enumerate(CELL_ORDER):
        x, y = (i % 2) * CELL_W, (i // 2) * CELL_H
        img = cell_imgs.get(name)
        if img is not None:
            im = Image.fromarray(img).convert("RGB")
            im.thumbnail((CELL_W - 8, CELL_H - 8), Image.LANCZOS)
            canvas.paste(im, (x + 4, y + 4))
        else:
            draw.rectangle([x + 4, y + 4, x + CELL_W - 4, y + CELL_H - 4],
                           fill=(24, 26, 27))
        draw.text((x + 8, y + CELL_H - 16), CELL_TITLES[name],
                  fill=(156, 154, 151))
    return np.asarray(canvas)
