# INTEGRATION.md — live-mode seams

Written before any live-mode code, per milestone spec. This documents how the
**existing, validated** pipeline is actually wired, where live mode plugs in,
and what must NOT change.

## 1. The validated stack (do not touch its behavior)

```
run_batch.py / server/app.py
    └─ runner/batch.py: build_stack(suite_name, checkpoint)
         ├─ sim/wrapper.py:      LiberoWrapper(suite_name, render_size=256, settle_steps=10, env_seed=0)
         └─ policy/octo_policy.py: OctoPolicy(checkpoint)
```

- **Env creation** (`sim/wrapper.py:99` `_make_env`): LIBERO `OffScreenRenderEnv`
  built from the task BDDL file, `camera_heights=camera_widths=render_size`.
  `env.seed(env_seed)` called once at creation (harness convention — layout RNG).
  LIBERO's `OffScreenRenderEnv` (`LIBERO/libero/libero/envs/env_wrapper.py:152`)
  is a thin `ControlEnv` over a robosuite 1.4.x env (mujoco 2.3.2 bindings).
  `camera_heights`/`camera_widths` are independent params → **non-square
  480×640 rendering is supported natively**; no LIBERO patch needed.
- **Reset** (`sim/wrapper.py:120`): initial object poses pinned via
  `benchmark.get_task_init_states(task_id)[seed % 50]` + `env.set_init_state`,
  then 10 no-op settle steps (`DUMMY_ACTION`, gripper open). Episodes are
  repeatable per (task, seed).
- **Step** (`sim/wrapper.py:134`): `env.step(action)` — action is 7-DoF
  (6 arm axis-angle delta + gripper), OSC_POSE controller, control_freq=20 Hz.
- **Obs** (`sim/wrapper.py:151` `obs()`): dict with exactly the keys the policy
  expects —
  - `image_primary` ← raw `agentview_image`, **rotated 180°**, JPEG-round-trip +
    lanczos3 resize to 256×256
  - `image_wrist` ← raw `robot0_eye_in_hand_image`, rotated 180°, resized 128×128
  - `proprio` ← eef_pos + quat2axisangle(eef_quat) + gripper_qpos (8-D)
  - `instruction` ← task language string
- **Render** (`sim/wrapper.py:140` `render()`): HWC uint8 of a raw sim camera
  image, **rotated 180°** — the same orientation the policy and the user see.
  THE 180° ROTATION IS THE CRITICAL CONVENTION FOR LIVE MODE (see §4).
- **Policy** (`policy/octo_policy.py`): `OctoPolicy.reset(instruction)` then
  `act(obs)` → (4,7) action chunk, receding-horizon. Action de-normalization
  and camera-key/resize conventions live in `sim/wrapper.py` + the policy
  wrapper and are the validated surface.
- **Closed loop** (`runner/run_episode.py`): obs → policy → execute chunk of
  K steps → replan; `check_success()` per step.

Live mode adds NEW code (`live/`) that REUSES `LiberoWrapper` and
`OctoPolicy` unchanged. The only difference: the live session constructs its
env at **480×640 native render** (user decision) instead of 256². The obs
pipeline resizes to the policy's 256²/128² exactly as before — policy input
is bit-identical in shape and preprocessing.

## 2. Where camera parameters live (for click unprojection)

All obtained from the simulator at runtime — nothing hardcoded:

- `wrapper.env` → `ControlEnv`; `.sim` → robosuite `MjSim` wrapping mujoco
  2.3.2 (`MjModel`/`MjData`).
- Camera id: `mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")`.
- Intrinsics: `model.cam_fovy[cam_id]` (vertical fov, degrees). Focal length
  in pixels: `f = (H/2) / tan(fovy/2)` with H the rendered image height.
- Extrinsics: `data.cam_xpos[cam_id]` (3,), `data.cam_xmat[cam_id]` (3×3,
  row-major, camera→world rotation). MuJoCo camera frame: **x = right,
  y = up, z = backwards** (camera looks down −z).
- Table plane: read from the sim at runtime (`main_table` body), not from a
  constant.

## 3. Where objects and impulses live (for perturbation)

- LIBERO `BDDLBaseDomain` (`LIBERO/libero/libero/envs/bddl_base_domain.py`)
  keeps `objects` (list of robosuite `MujocoObject`), `objects_dict`, and
  `obj_body_id` (name → body id), skipping `main_table`/`floor`/etc.
- Each movable object root body has a **free joint**; velocity address via
  `mujoco.mj_name2id(model, mjtObj.mjOBJ_JOINT, joint_name)` →
  `model.jnt_dofadr[jnt_id]` (qvel) / `model.jnt_qposadr` (qpos).
- Perturb = **add a velocity impulse** to the free joint's linear qvel
  components (`data.qvel[dofadr:dofadr+3] += impulse`), NOT a qpos write.
  Physics dissipates it naturally. Clicks whose ray-table intersection has no
  object within a sane radius are ignored.
- Object world positions: `data.body_xpos[body_id]` (or
  `data.get_body_xpos(name)`).

## 4. THE 180° rotation caveat (classic mirrored-axis bug lives here)

Raw mujoco/robosuite camera images are rotated 180° by `LiberoWrapper`
(`_rotate180`) before ANY consumer (policy obs, mp4 frames, live stream) sees
them. Therefore:

- The live stream shows the **rotated** image (the upright view the user
  already knows from the M0/M1/M2 artifacts).
- A click at displayed pixel (x, y) maps to **raw** camera pixel
  (W−1−x, H−1−y) before unprojection.
- Whether the raw image's pixel (u, v) corresponds to camera-frame
  (+u → +x_cam, +v → −y_cam) or a flip thereof depends on whether
  robosuite flipud's the mujoco buffer (it does — `MjSim.render` returns the
  buffer bottom-up). **This is verified empirically via the debug round-trip
  (§5), never assumed from reading the pipeline.**

## 5. Live-mode architecture (new code, one session / one viewer / localhost)

```
live/app.py            standalone FastAPI app, port 7870 (separate process
                       from the queued M1/M2 server on 7860 — no shared state)
  ├─ LiveSession       owns its own LiberoWrapper(render at 480x640) +
  │                    OctoPolicy via build_stack; runs the closed loop in an
  │                    executor thread: obs → policy → K steps → replan.
  │                    start(task_id, seed, instruction?) / reset / stop.
  ├─ GET /             serves live/index.html
  ├─ GET /tasks        task list (suite tasks, same source as M0 supports)
  ├─ WS /ws            newest-frame-only JPEG stream @ ~10 fps
  │                    (per-client latest-slot, never a backlog queue);
  │                    server logs fps, frame age, drops, policy latency
  ├─ POST /click {x, y, debug}  → unproject via §2/§4 → optional impulse;
  │                    with debug=1 returns the 3D point reprojected back to
  │                    displayed-image pixels for the on-screen dot overlay
  └─ GET /session      status snapshot
live/index.html        ~100 lines vanilla JS: img stream, task/instruction
                       controls, start/reset/stop, click-to-push,
                       debug-dot overlay ON by default, fps/latency readout
```

Behavior notes per the milestone contract:
- Session runs for minutes; episode `check_success` does not auto-stop it —
  the user resets/stops. (Success state is surfaced in the status endpoint.)
- Newest-frame-only: each WS client has a single-slot mailbox; the producer
  overwrites, the consumer drains. Stale frames are dropped and counted.
- Debug mode is the trust gate for the whole perturb path: click → 3D point
  → reproject → dot. No coordinate-frame debugging from logs.

## 6. Explicitly unchanged

- `sim/wrapper.py`, `policy/octo_policy.py`, `runner/`, `server/` (M1/M2),
  `run_batch.py` — untouched. Live mode imports them; it does not patch them.
- No new dependencies (websockets/fastapi/uvicorn/pillow already in the venv).
