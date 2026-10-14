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
    "path": "plan view (top-down)",
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

    def path(self, plan_xy, trail_xy, eef_xy):
        fig = self._fig("path")
        ax = self._ax(fig, "plan view, top-down (world xy, m)")
        if trail_xy is not None and len(trail_xy) > 1:
            t = np.asarray(trail_xy)
            ax.plot(t[:, 0], t[:, 1], "-", color="#57ab5a", lw=2,
                    label="executed")
        if plan_xy is not None and len(plan_xy) > 1:
            p = np.asarray(plan_xy)
            ax.plot(p[:, 0], p[:, 1], ".-", color="#539bf5", lw=1.5,
                    ms=3, label="imagined plan (50)")
        if eef_xy is not None:
            ax.plot(*eef_xy, "o", color="#e05d38", ms=7, label="gripper")
        ax.legend(fontsize=6, loc="best")
        ax.set_aspect("equal")
        return self._to_rgb(fig)

    def denoise(self, trajs_xy):
        fig = self._fig("denoise")
        ax = self._ax(fig, "denoising: 10 refinements of the plan (world xy)")
        n = len(trajs_xy)
        for i, xy in enumerate(trajs_xy):
            if len(xy) < 2:
                continue
            t = np.asarray(xy)
            alpha = 0.15 + 0.85 * (i / max(n - 1, 1))
            ax.plot(t[:, 0], t[:, 1], "-", color=(0.86, 0.31, 0.78, alpha),
                    lw=2 if i == n - 1 else 1)
        ax.set_aspect("equal")
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
