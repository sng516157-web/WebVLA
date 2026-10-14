"""Action-chunk -> image projection for X-Ray overlays.

The action vector is [dx, dy, dz, dax, day, daz, gripper]: pose DELTAS.
The frame those deltas live in is NOT assumed — setup/xray_waypoint_check.py
verifies empirically (predicted next EE position vs actual after stepping)
which convention holds, and FRAME_CONVENTION here must match its verdict.
"""

import numpy as np

# Verified by setup/xray_waypoint_check.py: LIBERO's OSC_POSE deltas are in
# the robot BASE frame; for the LIBERO table setups base == world identity
# (see the check's output). If a suite ever rotates the base, apply the base
# rotation here.
FRAME_CONVENTION = "world"  # "world" | "base"


def chunk_waypoints_world(full_chunk, eef_pos, base_rot=None):
    """Integrate the position deltas of a (N, 7) chunk into world waypoints.

    Returns (N, 3) world positions, starting from eef_pos after one step.
    """
    deltas = np.asarray(full_chunk, dtype=np.float64)[:, :3]
    if FRAME_CONVENTION == "base" and base_rot is not None:
        deltas = deltas @ np.asarray(base_rot).T
    return np.asarray(eef_pos, dtype=np.float64)[None, :] + np.cumsum(deltas, axis=0)


def world_to_px(cam, pts):
    """Project (N, 3) world points to display pixels; drops points behind
    the camera. Returns ([[x, y], ...], kept_mask)."""
    out, keep = [], []
    for p in pts:
        uv = cam.world_to_display(p)
        keep.append(uv is not None)
        out.append([float(uv[0]), float(uv[1])] if uv is not None else [None, None])
    return out, keep


def downsample(pts, n=25):
    """Evenly subsample a polyline to <= n points (keeps endpoints)."""
    if len(pts) <= n:
        return pts
    idx = np.linspace(0, len(pts) - 1, n).round().astype(int)
    return [pts[i] for i in idx]
