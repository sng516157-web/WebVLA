"""X-Ray engine: per-replan-tick introspection for a LiveSession.

Attached to LiveSession as `session.xray`. All computation happens on replan
ticks, off the policy->sim critical path (tensors are observed during the
policy call; everything else runs after the chunk is in hand). When no
overlay is enabled the episode loop takes the untouched act() path.

Overlays (each independently toggleable via the "xray.toggle" state-intent):
  waypoints  - full-chunk EE trajectory projected onto the stream
  bars       - the 7-D action vector being executed right now
  timeline   - replan ticks vs execution (client-rendered from xray.chunk)
  denoise    - the 10 flow-matching trajectory estimates converging
  heatmap    - instruction-token -> image-patch attention (16x16)
  bundle     - ring buffer of the last 5 replans, flushed on failure
"""

import numpy as np

from .bundle import ReplanRingBuffer
from .projection import chunk_waypoints_world, downsample, world_to_px

OVERLAYS = ("waypoints", "bars", "timeline", "denoise", "heatmap", "bundle")

# internals the policy must capture for each overlay
_WANT = {
    "denoise": {"denoise"},
    "heatmap": {"attn_prefix"},
    "bundle": {"denoise", "attn_prefix"},
}


class XRayEngine:
    def __init__(self, emit):
        self.emit = emit  # fn(msg_type, payload) -> broadcast envelope
        self.enabled = set()
        self.ring = ReplanRingBuffer()
        self.replan_idx = -1

    @property
    def active(self):
        return bool(self.enabled)

    def toggle(self, overlay, enabled):
        if overlay not in OVERLAYS:
            raise ValueError(f"unknown overlay {overlay!r}")
        (self.enabled.add if enabled else self.enabled.discard)(overlay)
        self.emit("session.state", {"xray": sorted(self.enabled)})

    def want(self):
        w = set()
        for ov in self.enabled:
            w |= _WANT.get(ov, set())
        return tuple(w)

    # ------------------------------------------------------- episode hooks
    def on_replan(self, session, chunk, full_chunk, internals, latency_ms):
        self.replan_idx += 1
        r = self.replan_idx
        cam = session._cam
        eef = np.asarray(session.wrapper._raw["robot0_eef_pos"],
                         dtype=np.float64)
        frame = None
        if "bundle" in self.enabled or "heatmap" in self.enabled:
            frame = session.last_frame_rgb  # set by the episode loop

        waypoints_px = None
        if full_chunk is not None and (
                {"waypoints", "timeline", "denoise", "bundle"} & self.enabled):
            wps = chunk_waypoints_world(full_chunk, eef)
            px, _ = world_to_px(cam, wps)
            waypoints_px = downsample([p for p in px if p[0] is not None])

        if {"waypoints", "timeline", "bars"} & self.enabled:
            self.emit("xray.chunk", {
                "replan": r, "latency_ms": round(latency_ms, 1),
                "exec_len": len(chunk),
                "chunk_len": int(len(full_chunk)) if full_chunk is not None
                else len(chunk),
                "waypoints_px": waypoints_px,
            })

        if "denoise" in self.enabled and internals.get("denoise_xt") is not None:
            trajs = []
            for xt in internals["denoise_xt"]:
                px, _ = world_to_px(cam, chunk_waypoints_world(xt, eef))
                trajs.append(downsample([p for p in px if p[0] is not None]))
            self.emit("xray.denoise", {"replan": r, "trajectories_px": trajs})

        if "heatmap" in self.enabled:
            for view, key in (("agentview", "heatmap_agentview"),
                              ("wrist", "heatmap_wrist")):
                grid = internals.get(key)
                if grid is not None:
                    self.emit("xray.heatmap", {
                        "replan": r, "view": view, "cols": 16,
                        "grid": [round(float(v), 4) for v in grid],
                    })

        if "bundle" in self.enabled:
            self.ring.push(r, full_chunk, internals, frame, {
                "latency_ms": round(latency_ms, 1),
                "task_id": session.task_id, "seed": session.seed,
                "instruction": session.instruction,
            })

    def on_step(self, k, action):
        if {"bars", "waypoints", "timeline"} & self.enabled:
            self.emit("xray.exec", {
                "replan": self.replan_idx, "k": k,
                "action": [round(float(v), 5) for v in action],
            })
        if "bundle" in self.enabled:
            self.ring.record_executed(action)

    def on_episode_end(self, session, success, error=None):
        if "bundle" in self.enabled and not success and self.ring.slots:
            path = self.ring.flush(
                reason="error" if error else "stopped_without_success",
                extra_meta={"task_id": session.task_id, "seed": session.seed,
                            "instruction": session.instruction,
                            "error": (error or "")[-500:] or None})
            self.emit("xray.bundle", {"replan": self.replan_idx,
                                      "path": path, "kept": len(self.ring.slots)})
            print(f"[xray] failure bundle saved -> {path}", flush=True)
        self.ring.clear()
        self.replan_idx = -1
