"""Live-mode session: one continuous closed-loop episode in a thread.

Reuses the validated stack unchanged:
  - sim.wrapper.LiberoWrapper (here constructed with render_size=640 for a
    native-res stream; the policy obs pipeline resizes to 256^2/128^2 exactly
    as in M0 — same FOV, no aspect distortion)
  - policy: OctoPolicy (default) or Pi05Policy via WEBVLA_POLICY=pi05
    (act -> action chunk, receding horizon; chunk length is the policy's)

Streamed frames: agentview raw (bottom-up rows) -> rotate 180 (the canonical
upright view) -> center-crop 640x640 -> 640x480 -> JPEG.

Click unprojection (verified against sim ground truth in live/_probe.py):
  raw pixel (u,v) has rows bottom-up, columns left-to-right in camera +x.
  displayed (xd,yd) -> raw u = R-1-xd, v = R-1-(yd+CROP_Y0)
  camera ray dir = [ (u-R/2)/f, (v-R/2)/f, -1 ] in camera frame (x right,
  y up, z back), world dir = cam_xmat @ dir, origin = cam_xpos.
  Intersect with table plane z = table_top_z (read from the sim at runtime).
Perturb = velocity impulse on the object's free joint (qvel), never qpos.
"""

import io
import queue
import threading
import time

import numpy as np
from PIL import Image

from policy.octo_policy import OctoPolicy
from sim.wrapper import LiberoWrapper


def _build_policy(backend, checkpoint):
    if backend == "pi05":
        from policy.pi0_policy import Pi05Policy

        return Pi05Policy()
    if backend == "lerobot_pi05":
        from policy.lerobot_pi05_policy import LerobotPi05Policy

        return LerobotPi05Policy()
    return OctoPolicy(checkpoint)

RENDER = 640                      # square native render
STREAM_W, STREAM_H = 640, 480     # stream = center crop of the rotated render
CROP_Y0 = (RENDER - STREAM_H) // 2

CONTROL_HZ = 20                   # robosuite control_freq; pace sim to real time
JPEG_QUALITY = 80

TABLE_BODY_CANDIDATES = ("table", "main_table", "countertop", "coffee_table")
PUSH_RADIUS_M = 0.12              # clicks farther than this from any object are ignored
PUSH_SPEED_MPS = 1.0              # planar velocity impulse magnitude


class FrameHub:
    """Newest-frame-only fanout. One shared slot; slow clients skip frames.

    Also fans out JSON event envelopes (X-Ray): each subscribed client's
    asyncio queue gets a thread-safe put; the ws handler drains it.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self.frame_id = 0
        self.jpeg = None
        self.ts = 0.0
        self.clients = {}  # id -> stats dict
        self._event_subs = {}  # id -> (asyncio loop, asyncio.Queue)

    def publish(self, jpeg, ts):
        with self._cond:
            self.frame_id += 1
            self.jpeg = jpeg
            self.ts = ts
            self._cond.notify_all()

    def wait_newer(self, last_id, timeout=2.0):
        with self._cond:
            self._cond.wait_for(lambda: self.frame_id > last_id, timeout=timeout)
            return self.frame_id, self.jpeg, self.ts

    def register(self):
        with self._cond:
            cid = len(self.clients)
            self.clients[cid] = {"sent": 0, "dropped": 0, "age_sum": 0.0, "age_max": 0.0}
            return cid

    def unregister(self, cid):
        with self._cond:
            self.clients.pop(cid, None)
            self._event_subs.pop(cid, None)

    def register_events(self, cid, loop, q):
        with self._cond:
            self._event_subs[cid] = (loop, q)

    def broadcast_event(self, msg_type, payload):
        """Encode an envelope and fan it out to subscribed clients."""
        from live import protocol

        text = protocol.encode(msg_type, payload, "server")
        with self._cond:
            subs = list(self._event_subs.values())
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, text)
            except RuntimeError:
                pass  # client loop already gone

    def client_stats(self):
        with self._cond:
            return {cid: dict(s) for cid, s in self.clients.items()}


class _Restart(Exception):
    def __init__(self, params):
        self.params = params


class CameraGeometry:
    """agentview intrinsics/extrinsics read from the sim; nothing hardcoded."""

    def __init__(self, wrapper, camera="agentview"):
        sim = wrapper.env.sim
        self.model = sim.model
        self.data = sim.data
        self.cam_id = self.model.camera_name2id(camera)

    def refresh(self):
        fovy = np.deg2rad(float(self.model.cam_fovy[self.cam_id]))
        self.f = (RENDER / 2.0) / np.tan(fovy / 2.0)
        self.pos = np.asarray(self.data.cam_xpos[self.cam_id], dtype=np.float64)
        self.mat = np.asarray(self.data.cam_xmat[self.cam_id], dtype=np.float64).reshape(3, 3)

    def display_to_ray(self, xd, yd):
        u = RENDER - 1.0 - xd
        v = RENDER - 1.0 - (yd + CROP_Y0)
        d_cam = np.array([(u - RENDER / 2.0) / self.f, (v - RENDER / 2.0) / self.f, -1.0])
        return self.pos, self.mat @ d_cam

    def world_to_display(self, p):
        pc = self.mat.T @ (np.asarray(p, dtype=np.float64) - self.pos)
        if pc[2] >= -1e-9:
            return None
        u = self.f * pc[0] / (-pc[2]) + RENDER / 2.0
        v = self.f * pc[1] / (-pc[2]) + RENDER / 2.0
        return (RENDER - 1.0 - u, RENDER - 1.0 - v - CROP_Y0)


def table_top_z(wrapper):
    sim = wrapper.env.sim
    model, data = sim.model, sim.data
    raw_model = model._model
    for name in TABLE_BODY_CANDIDATES:
        try:
            bid = model.body_name2id(name)
        except ValueError:
            continue
        top = 0.0
        for g in range(raw_model.ngeom):
            if raw_model.geom_bodyid[g] == bid and int(raw_model.geom_type[g]) == 6:  # box
                top = max(top, raw_model.geom_pos[g][2] + raw_model.geom_size[g][2])
        return float(data.body_xpos[bid][2] + top)
    raise RuntimeError(f"no support-surface body found among {TABLE_BODY_CANDIDATES}")


def movable_objects(wrapper):
    """[(name, body_id)] for the task's movable objects (fixtures excluded)."""
    inner = wrapper.env.env
    return [(o.name, inner.obj_body_id[o.name]) for o in inner.objects]


class LiveSession(threading.Thread):
    """Owns the env + policy and runs the episode loop off the asyncio loop."""

    def __init__(self, checkpoint, suite_name="libero_spatial", policy_backend="octo"):
        super().__init__(daemon=True, name="live-session")
        self.suite_name = suite_name
        self.checkpoint = checkpoint
        self.policy_backend = policy_backend
        self.wrapper = None
        self.policy = None
        self.hub = FrameHub()

        self.state = "idle"  # idle | running
        self.task_id = None
        self.seed = 0
        self.instruction = None
        self.success = False
        self.steps = 0
        self.episode_started_at = None
        self.replan_latencies = []
        self.pushes = 0

        self._cmd = queue.Queue()
        self._stop = threading.Event()
        self._restart = None  # dict params -> restart current episode
        self._lock = threading.Lock()  # guards env access for clicks
        self._cam = None
        self._table_z = None
        self.last_frame_rgb = None  # latest upright frame (for X-Ray bundle)
        from live.xray.engine import XRayEngine

        self.xray = XRayEngine(self.hub.broadcast_event)

    # ---- lifecycle ------------------------------------------------------

    def load(self):
        """Heavy init (env class + checkpoint). Call once before start()."""
        self.wrapper = LiberoWrapper(self.suite_name, render_size=RENDER)
        self.policy = _build_policy(self.policy_backend, self.checkpoint)

    def run(self):
        while True:
            cmd, params = self._cmd.get()
            if cmd == "shutdown":
                return
            if cmd == "start":
                try:
                    self._episode_loop(params)
                except Exception:
                    import traceback
                    tb = traceback.format_exc()
                    traceback.print_exc()
                    try:
                        self.xray.on_episode_end(self, success=False, error=tb)
                    except Exception:
                        pass
                    self.state = "idle"

    def request_start(self, task_id, seed, instruction=None):
        if self.state != "idle":
            raise RuntimeError("session already running; stop or reset first")
        self._cmd.put(("start", {"task_id": task_id, "seed": seed, "instruction": instruction}))

    def request_stop(self):
        self._stop.set()

    def request_reset(self, task_id=None, seed=None, instruction=None):
        if self.state == "idle" and self.task_id is None:
            raise RuntimeError("nothing to reset; start a session first")
        self._restart = {
            "task_id": self.task_id if task_id is None else task_id,
            "seed": self.seed if seed is None else seed,
            "instruction": self.instruction if instruction is None else instruction,
        }
        self._stop.set()
        if self.state == "idle":
            self._cmd.put(("start", self._restart))
            self._restart = None

    def shutdown(self):
        self._stop.set()
        self._cmd.put(("shutdown", None))

    # ---- episode loop ---------------------------------------------------

    def _episode_loop(self, params):
        while True:
            try:
                self._run_episode(params)
                return
            except _Restart as r:
                params = r.params

    def _run_episode(self, params):
        self.task_id = params["task_id"]
        self.seed = params["seed"]
        self.instruction = params.get("instruction")
        self.state = "resetting"  # env reset + settle takes a few seconds
        self.steps = 0
        self.success = False
        self._stop.clear()
        self._restart = None

        obs = self.wrapper.reset(self.task_id, self.seed)
        instruction = (self.instruction or "").strip() or obs["instruction"]
        self.policy.reset(instruction)
        with self._lock:
            self._cam = CameraGeometry(self.wrapper)
            self._cam.refresh()
            self._table_z = table_top_z(self.wrapper)
        self.state = "running"
        self.episode_started_at = time.time()
        self.replan_latencies = []
        print(f"[live] episode start: task={self.task_id} seed={self.seed} "
              f"instruction={instruction!r} table_z={self._table_z:.3f}", flush=True)

        last_log = time.time()
        last_log_steps = 0
        last_log_frames = self.hub.frame_id

        # Policy-specific obs view (Octo: obs(); pi05: obs_pi05()); both read
        # the latest sim state, equivalent to the obs returned by step().
        obs_fn = getattr(self.wrapper, getattr(self.policy, "OBS_METHOD", "obs"))
        xray = self.xray
        while not self._stop.is_set():
            t0 = time.time()
            if xray.active and hasattr(self.policy, "act_with_internals"):
                # READ-ONLY: proven equivalent to act() by
                # setup/xray_equivalence_check.py
                chunk, full_chunk, internals = self.policy.act_with_internals(
                    obs_fn(), want=xray.want())
            else:
                chunk = self.policy.act(obs_fn())
                full_chunk, internals = chunk, {}
            lat_ms = (time.time() - t0) * 1000.0
            self.replan_latencies.append(lat_ms)
            if xray.active:
                with self._lock:
                    self._cam.refresh()
                xray.on_replan(self, chunk, full_chunk, internals, lat_ms)
            for k, action in enumerate(chunk):
                if self._stop.is_set():
                    break
                step_start = time.time()
                with self._lock:
                    if self._restart is not None:
                        raise _Restart(self._restart)
                    obs, _, _, _ = self.wrapper.step(action)
                    self.steps += 1
                    if self.wrapper.check_success():
                        self.success = True
                    frame = self.wrapper.render("agentview")
                crop = frame[CROP_Y0:CROP_Y0 + STREAM_H]
                self.last_frame_rgb = crop
                buf = io.BytesIO()
                Image.fromarray(crop).save(buf, format="JPEG", quality=JPEG_QUALITY)
                self.hub.publish(buf.getvalue(), time.time())
                if xray.active:
                    xray.on_step(k, action)
                # pace sim to real time (20 Hz control)
                elapsed = time.time() - step_start
                period = 1.0 / CONTROL_HZ
                if elapsed < period:
                    time.sleep(period - elapsed)

            now = time.time()
            if now - last_log >= 5.0:
                self._log_stats(now, now - last_log, self.steps - last_log_steps,
                                self.hub.frame_id - last_log_frames)
                last_log, last_log_steps, last_log_frames = now, self.steps, self.hub.frame_id

        if self._restart is not None:
            xray.on_episode_end(self, success=self.success)
            raise _Restart(self._restart)
        self.state = "idle"
        xray.on_episode_end(self, success=self.success)
        print(f"[live] episode stopped: task={self.task_id} seed={self.seed} "
              f"steps={self.steps} success={self.success}", flush=True)

    def _log_stats(self, now, window_s, window_steps, window_frames):
        lat = self.replan_latencies[-100:]
        mean_lat = float(np.mean(lat)) if lat else 0.0
        p95_lat = float(np.percentile(lat, 95)) if lat else 0.0
        parts = [
            f"[live] steps={self.steps} ({window_steps / window_s:.1f}/s)",
            f"frames_published={window_frames / window_s:.1f}/s",
            f"replan_ms mean={mean_lat:.0f} p95={p95_lat:.0f}",
        ]
        for cid, s in self.hub.client_stats().items():
            n = max(s["sent"], 1)
            parts.append(
                f"client{cid}: sent={s['sent']} dropped={s['dropped']} "
                f"age_ms mean={1000 * s['age_sum'] / n:.0f} max={1000 * s['age_max']:.0f}"
            )
        print(" | ".join(parts), flush=True)

    # ---- clicks ----------------------------------------------------------

    def handle_click(self, xd, yd, push=True):
        """Unproject a displayed-pixel click; optionally shove the nearest object.

        Safe to call from any thread: env access is serialized with the
        episode loop via self._lock (qvel writes land between sim steps).
        """
        if self.wrapper is None or self.wrapper.env is None or self._cam is None:
            raise RuntimeError("no scene yet; start a session first")
        with self._lock:
            self._cam.refresh()
            origin, direction = self._display_ray(xd, yd)
            hit = self._intersect_table(origin, direction)
            result = {
                "click": [xd, yd],
                "hit_world": None,
                "reprojected": None,
                "object": None,
                "pushed": False,
                "impulse": None,
                "reason": None,
            }
            if hit is None:
                result["reason"] = "ray does not intersect the table plane"
                return result
            result["hit_world"] = [round(float(c), 4) for c in hit]
            rp = self._cam.world_to_display(hit)
            result["reprojected"] = [round(rp[0], 1), round(rp[1], 1)] if rp else None
            if not push:
                return result

            obj = self._nearest_object(hit)
            if obj is None:
                result["reason"] = f"no object within {PUSH_RADIUS_M} m of the hit point"
                return result
            name, body_id, pos, dist = obj
            d = pos[:2] - hit[:2]
            n = float(np.linalg.norm(d))
            if n < 1e-3:
                # click dead-center on the object: shove away from the camera
                d = pos[:2] - origin[:2]
                n = float(np.linalg.norm(d)) or 1.0
            impulse = np.array([d[0] / n, d[1] / n, 0.0]) * PUSH_SPEED_MPS
            self._apply_impulse(name, impulse)
            self.pushes += 1
            result.update(
                object=name,
                object_dist_m=round(dist, 4),
                pushed=True,
                impulse=[round(float(c), 3) for c in impulse],
            )
            print(f"[live] push: {name} impulse={result['impulse']} "
                  f"hit={result['hit_world']} dist={dist:.3f}m", flush=True)
            return result

    def _display_ray(self, xd, yd):
        return self._cam.display_to_ray(xd, yd)

    def _intersect_table(self, origin, direction):
        if direction[2] >= -1e-9:
            return None
        t = (self._table_z - origin[2]) / direction[2]
        if t <= 0:
            return None
        return origin + t * direction

    def _nearest_object(self, hit):
        data = self.wrapper.env.sim.data
        best = None
        for name, body_id in movable_objects(self.wrapper):
            pos = np.asarray(data.body_xpos[body_id], dtype=np.float64)
            dist = float(np.linalg.norm(pos[:2] - hit[:2]))
            if dist <= PUSH_RADIUS_M and (best is None or dist < best[3]):
                best = (name, body_id, pos, dist)
        return best

    def _apply_impulse(self, object_name, dvel):
        sim = self.wrapper.env.sim
        qvel_adr = sim.model.get_joint_qvel_addr(f"{object_name}_joint0")[0]
        sim.data.qvel[qvel_adr:qvel_adr + 3] += dvel

    # ---- status ----------------------------------------------------------

    def status(self):
        lat = self.replan_latencies[-100:]
        uptime = time.time() - self.episode_started_at if self.episode_started_at else 0.0
        return {
            "state": self.state,
            "suite": self.suite_name,
            "task_id": self.task_id,
            "seed": self.seed,
            "instruction": self.instruction,
            "success": self.success,
            "steps": self.steps,
            "pushes": self.pushes,
            "episode_uptime_s": round(uptime, 1),
            "frames_published": self.hub.frame_id,
            "replan_ms": {
                "mean": round(float(np.mean(lat)), 1) if lat else None,
                "p95": round(float(np.percentile(lat, 95)), 1) if lat else None,
            },
            "stream": {"width": STREAM_W, "height": STREAM_H, "render": RENDER},
            "clients": self.hub.client_stats(),
        }
