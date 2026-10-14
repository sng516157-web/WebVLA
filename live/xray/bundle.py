"""Failure introspection: keep the last N replans so failures are explorable.

The ring buffer records per-replan: the full predicted chunk, the executed
actions, the denoising trajectory and heatmaps (when captured), and the
rendered frame at the tick. On episode failure the bundle is flushed to
live/xray_bundles/<timestamp>/ (gitignored).
"""

import json
import os
import time
from collections import deque

import numpy as np

import paths

BUNDLES_DIR = os.path.join(paths.REPO_ROOT, "live", "xray_bundles")
KEEP_REPLANS = 5


class ReplanRingBuffer:
    def __init__(self, keep=KEEP_REPLANS):
        self.keep = keep
        self.slots = deque(maxlen=keep)

    def push(self, replan, full_chunk, internals, frame_rgb, meta):
        self.slots.append({
            "replan": replan,
            "full_chunk": np.asarray(full_chunk, dtype=np.float32)
            if full_chunk is not None else None,
            "denoise_xt": internals.get("denoise_xt") if internals else None,
            "heatmap_agentview": internals.get("heatmap_agentview")
            if internals else None,
            "heatmap_wrist": internals.get("heatmap_wrist")
            if internals else None,
            "frame_rgb": frame_rgb,
            "meta": dict(meta),
            "executed": [],
        })

    def record_executed(self, action):
        if self.slots:
            self.slots[-1]["executed"].append(
                np.asarray(action, dtype=np.float32))

    def clear(self):
        self.slots.clear()

    def flush(self, reason, extra_meta=None):
        """Write the buffered replans to disk. Returns the bundle dir."""
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out = os.path.join(BUNDLES_DIR, ts)
        os.makedirs(out, exist_ok=True)
        index = {"reason": reason, "n_replans": len(self.slots),
                 "slots": [], **(extra_meta or {})}
        from PIL import Image

        for i, s in enumerate(self.slots):
            name = f"replan{s['replan']:03d}"
            arrays = {k: s[k] for k in
                      ("full_chunk", "denoise_xt", "heatmap_agentview",
                       "heatmap_wrist") if s[k] is not None}
            if s["executed"]:
                arrays["executed"] = np.stack(s["executed"])
            np.savez_compressed(os.path.join(out, name + ".npz"), **arrays)
            if s["frame_rgb"] is not None:
                Image.fromarray(s["frame_rgb"]).save(
                    os.path.join(out, name + ".jpg"), quality=85)
            index["slots"].append({"replan": s["replan"], "file": name,
                                   **s["meta"]})
        with open(os.path.join(out, "index.json"), "w") as f:
            json.dump(index, f, indent=2)
        return out
