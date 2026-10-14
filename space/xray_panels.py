"""Separate-panel renderers for Space X-Ray (matplotlib Agg -> uint8 HWC).

Same data as space/xray_overlay.py, but rendered as standalone plots so the
video stays clean. Figures are reused across calls within an episode
(clear + redraw) — a few ms per refresh, off the policy->sim critical path.

Honesty labels are in the UI captions; plots additionally carry a short
"factual" title only.
"""

import matplotlib

matplotlib.use("Agg")

import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_BAR_LABELS = ["dx", "dy", "dz", "dax", "day", "daz", "grip"]
_BAR_SCALE = np.array([0.05, 0.05, 0.05, 0.35, 0.35, 0.35, 1.0])


class PanelRenderer:
    def __init__(self):
        self._figs = {}

    def _fig(self, key, size=(2.8, 2.2)):
        if key not in self._figs:
            fig = plt.figure(figsize=size, dpi=100)
            self._figs[key] = fig
        fig = self._figs[key]
        fig.clear()
        return fig

    @staticmethod
    def _to_rgb(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"))

    def close(self):
        for fig in self._figs.values():
            plt.close(fig)
        self._figs.clear()

    # ------------------------------------------------------------- panels
    def bars(self, action):
        """The 7-D action executing right now."""
        fig = self._fig("bars")
        ax = fig.add_subplot()
        a = np.asarray(action, dtype=float)
        colors = ["#57ab5a" if v >= 0 else "#e5534b" for v in a]
        ax.bar(range(7), a, color=colors)
        ax.set_xticks(range(7), _BAR_LABELS, fontsize=7)
        ax.axhline(0, color="#373a3d", lw=0.8)
        lim = float(np.max(np.abs(a))) or 1e-6
        ax.set_ylim(-1.2 * lim, 1.2 * lim)
        ax.set_title("action being executed (7-D)", fontsize=8)
        ax.tick_params(labelsize=7)
        return self._to_rgb(fig)

    def timeline(self, ticks, cur):
        """One bar per replan tick; height = inference ms; orange = running."""
        fig = self._fig("timeline")
        ax = fig.add_subplot()
        n = len(ticks)
        show = list(range(max(0, n - 40), n))
        colors = ["#e05d38" if i == cur else "#539bf5" for i in show]
        ax.bar(show, [ticks[i] for i in show], color=colors)
        ax.set_xlabel("replan tick", fontsize=7)
        ax.set_ylabel("inference ms", fontsize=7)
        ax.set_title("replan timeline (orange = executing)", fontsize=8)
        ax.tick_params(labelsize=7)
        return self._to_rgb(fig)

    def path(self, plan_xy, trail_xy, eef_xy):
        """Top-down (world XY) view: the imagined 50-step plan vs the path
        actually executed so far this episode."""
        fig = self._fig("path")
        ax = fig.add_subplot()
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
        ax.set_title("plan view, top-down (world xy, m)", fontsize=8)
        ax.tick_params(labelsize=6)
        return self._to_rgb(fig)

    def denoise(self, trajs_xy):
        """The 10 flow-matching estimates of the same plan, faint -> solid."""
        fig = self._fig("denoise")
        ax = fig.add_subplot()
        n = len(trajs_xy)
        for i, xy in enumerate(trajs_xy):
            if len(xy) < 2:
                continue
            t = np.asarray(xy)
            alpha = 0.15 + 0.85 * (i / max(n - 1, 1))
            ax.plot(t[:, 0], t[:, 1], "-", color=(0.86, 0.31, 0.78, alpha),
                    lw=2 if i == n - 1 else 1)
        ax.set_aspect("equal")
        ax.set_title("denoising: 10 refinements of the plan (world xy)",
                     fontsize=8)
        ax.tick_params(labelsize=6)
        return self._to_rgb(fig)

    def heatmap(self, grid):
        """16x16 instruction->image attention, standalone (agentview)."""
        fig = self._fig("heatmap", size=(2.4, 2.2))
        ax = fig.add_subplot()
        ax.imshow(np.asarray(grid, dtype=float).reshape(16, 16),
                  cmap="hot", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("instruction attention over image patches", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        return self._to_rgb(fig)
