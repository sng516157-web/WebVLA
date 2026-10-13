# WebVLA — sim-based VLA demo (LIBERO + Octo)

M0 (below): local Python pipeline — natural-language instruction → Octo policy
in closed loop on LIBERO simulation tasks → per-episode mp4 + metadata JSON.
M1 (bottom of this file): the same validated loop behind a hosted queue-mode
web demo (FastAPI + Gradio, deployable to HF Spaces).

## Layout

- `paths.py` — **all machine-specific paths live here**, overridable via
  `WEBVLA_HOME`, `WEBVLA_LIBERO`, `WEBVLA_OCTO`, `WEBVLA_VENV`, `WEBVLA_RLDS`,
  `WEBVLA_CKPTS` env vars (defaults match the original install: `/root/webvla/...`)
- `sim/wrapper.py` — thin LIBERO env wrapper: `reset(task_id, seed)`, `step(action)`,
  `render(camera_name) -> HWC uint8`, `obs() -> dict` (policy-keyed images, proprio, instruction)
- `policy/octo_policy.py` — Octo checkpoint loading + windowed action-chunk prediction
- `runner/run_episode.py` — closed-loop episode generator (yields frames) + mp4/metadata saver
- `runner/batch.py` — shared grid runner, sanity check, aggregation
- `reference_eval.py` — step 1: reference validation (2 tasks × 3 episodes)
- `run_batch.py` — step 4: batch run (3 tasks × 5 seeds) → `episodes/`, `results.json`
- `setup/` — environment bootstrap scripts (all honor the `WEBVLA_*` overrides)

## Environment (original machine)

The stack runs inside **WSL2 Ubuntu 26.04** — native Windows is not viable
(JAX CUDA wheels are Linux-only); any Linux box with a CUDA GPU should work.
Third-party code is pinned: LIBERO master @ 8f1084e, octo main @ 241fb35,
checked out under `$WEBVLA_HOME` (default `/root/webvla/{LIBERO,octo}`).

Pinned core versions (verified on disk):

| package | version | note |
|---|---|---|
| python | 3.10.22 | uv-managed CPython |
| mujoco | 2.3.2 | LIBERO pin |
| robosuite | 1.4.0 | LIBERO pin |
| jax / jaxlib | 0.4.20 / 0.4.20+cuda12.cudnn89 | octo pin |
| flax | 0.7.5 | octo pin |
| tensorflow | 2.15.0 (CPU-visible only) | octo pin |
| numpy | 1.24.3 | **deviation**: LIBERO pins 1.22.4, but TF 2.15 needs ≥1.23.5 and octo pins 1.24.3 |
| matplotlib | 3.7.5 | **deviation**: LIBERO pins 3.5.3, broken by numpy 1.24 |
| einops | 0.6.1 | **deviation**: octo needs ≥0.6.1, LIBERO pins 0.4.1 (training-only) |
| transformers | 4.34.1 | **deviation**: octo needs ≥4.34.1 (T5 tokenizer), LIBERO pins 4.21.1 (training-only) |
| torch | 2.5.1+cpu | unpinned by LIBERO; needed only for `torch.load` of init states; ≥2.6 flips `weights_only` and breaks it |
| scipy | 1.11.4 | jax 0.4.20 needs `scipy.linalg.tril`, removed in scipy 1.15 |

Platform fixes applied (documented, no dep changes):
- `patchelf --clear-execstack` on jaxlib `.so`s — Ubuntu 26.04 glibc refuses
  executable-stack libraries (jaxlib 0.4.20 predates the policy change).
- `nvidia-cudnn-cu12==8.9.2.26` forced — jax's `cuda12_pip` extra allows cudnn≥8.9
  and resolves to 9.x, which jaxlib 0.4.20 (cudnn89 build) cannot load.
- LIBERO editable install replaced by a `.pth` (PEP 660 mapping came out empty).
- WSL networking (original machine only): a Windows proxy was reachable via
  WSL VirtioProxy localhost forwarding. If you are behind a proxy, write your
  own `http_proxy`/`https_proxy` exports to `$HOME/webvla-env.sh` — all setup
  scripts source it if present (optional; not needed on a direct connection).
- Headless rendering: `MUJOCO_GL=egl` verified working (OSMesa verified as fallback;
  use `MUJOCO_GL=osmesa bash setup/run.sh ...`).

## How to run

All commands run **inside Linux** (WSL2 or native) with the repo as CWD —
`setup/run.sh` locates the repo from its own path, sets `PYTHONPATH`, activates
the venv at `$WEBVLA_VENV`, and exports `MUJOCO_GL=egl`. Set the `WEBVLA_*`
env vars first if your third-party install is not at `/root/webvla`.

```bash
# one-time setup (as root on a fresh WSL2/Linux box)
bash setup/wsl_setup_env.sh && bash setup/wsl_post_install.sh \
  && bash setup/wsl_fix_stack.sh && bash setup/wsl_fix_execstack.sh \
  && bash setup/wsl_fix_numpy.sh && bash setup/wsl_fix2.sh

# reference validation (2 tasks x 3 episodes; --checkpoint defaults to octo-base)
bash setup/run.sh reference_eval.py \
  --checkpoint $WEBVLA_CKPTS/octo_finetune/libero_spatial/<your_finetune_run> \
  --out-dir reference/ft50000

# batch (3 tasks x 5 seeds) -> episodes/ + results.json
bash setup/run.sh run_batch.py \
  --checkpoint $WEBVLA_CKPTS/octo_finetune/libero_spatial/<your_finetune_run> \
  --reference reference/ft50000/reference_results.json

# fine-tune from scratch (~5h): downloads dataset, converts to RLDS, trains
bash setup/wsl_download_data.sh   # libero_spatial hdf5s (curl resume)
bash setup/run.sh -m tools.convert_libero_to_rlds --suite libero_spatial
bash setup/run_finetune.sh
```

On Windows, prefix each command with
`wsl -d Ubuntu --` and use the repo's `/mnt/c/...` path for the script.

## Results

All numbers below are read from files on disk (`reference/**/reference_results.json`,
`results.json`), never from memory.

**Step-1 reference eval (2 tasks × 3 seeds, closed loop):**

| policy | task 0 | task 1 | overall | source |
|---|---|---|---|---|
| octo-base 1.5 (zero-shot) | 0/3 | 0/3 | **0/6 = 0%** | `reference/reference_results.json` |
| octo fine-tuned (this repo, 50k steps) | 3/3 | 2/3 | **5/6 = 83.3%** | `reference/ft50000/reference_results.json` |

Published comparison point: Octo fine-tuned scores **78.9%** on LIBERO-Spatial
(10 tasks × 50 episodes, OpenVLA paper Table 3 — no official Octo LIBERO eval
script exists; verified across octo repo main/main_old/tags). Our 2-task
reference eval (83.3%) is consistent with that number.

**Step-4 batch (3 tasks × 5 seeds) — `results.json`:**

| task | success | instruction |
|---|---|---|
| 0 | 3/5 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| 1 | 4/5 | pick up the black bowl next to the ramekin and place it on the plate |
| 2 | 4/5 | pick up the black bowl from table center and place it on the plate |
| **overall** | **11/15 = 73.3%** | |

- Inference latency per replan (batch): mean 157ms, p50 156ms, p95 160ms
  (max 7.5s = first-call JIT compile), on the RTX 4080 Laptop.
- Camera-key sanity check (printed by `run_batch.py`, stored in
  `results.json.sanity_check`): obs `image_primary` (256,256,3) and
  `image_wrist` (128,128,3) match the checkpoint's `example_batch` exactly —
  **MATCH**. (`proprio` is fed but not consumed by this checkpoint.)
- Failed seeds: `task0/seed1`, `task0/seed2`, `task1/seed4`, `task2/seed4`.
  Video inspection of `task0/seed1`: the arm reaches and grasps the correct
  bowl, but it slips from the gripper mid-transport — a grasp-physics fumble
  on an otherwise correct policy, i.e. a genuinely flaky seed rather than a
  systematic failure.

## Fine-tuning (why and how)

octo-base zero-shot scored 0/6 — published LIBERO numbers are for the
**fine-tuned** model, so we fine-tuned (user decision):

1. Dataset: 10 libero_spatial hdf5 files (500 demos) from
   `huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets` → RLDS via
   `tools/convert_libero_to_rlds.py` (images rotated 180° to upright,
   state = ee_pos+ee_ori+gripper_states, actions raw).
2. Config `setup/finetune_libero_config.py`: octo `finetune.py` defaults
   except batch 256→16 (12GB VRAM), **lr peak 3e-4→5e-5** (see below),
   warmup 500, 50k steps, shuffle buffer 500 + 6 frame threads (host RAM).
3. Run: `bash setup/run_finetune.sh` (~5h at ~2.5 it/s on the 4080 Laptop;
   checkpoints every 5k in `$WEBVLA_CKPTS/octo_finetune/...`).

**The LR finding (documented deviation):** octo's default peak LR 3e-4 is
tuned for batch 256; at batch 16 the run diverged — action-prediction MSE on
held-out demos was 0.67 (worse than predicting the mean; GT variance ≈ 0.2)
and the policy scored 0/6. At LR 5e-5 the MSE dropped to 0.079 after only
3k steps. Diagnostic: `setup/check_action_mse.py <checkpoint_dir>`.
Input conventions were verified before blaming training: image orientation
(`setup/check_orientation.py`), proprio eval-vs-train match
(`setup/check_proprio.py`), language mapping, and action stats.

Eval-side proprio note: the finetuned checkpoint's `example_batch` contains
`proprio`, and the wrapper feeds normalized proprio, but the sanity check
reports the policy does not consume it — octo-base is image+language only.
This is consistent between training and eval and costs nothing.

## Known flaky seeds

From `results.json.failed_episodes` (batch run, finetuned 50k ckpt):
`task0/seed1` (grasp slips mid-transport), `task0/seed2`, `task1/seed4`,
`task2/seed4`. Reference eval (seeds 0-2, tasks 0-1) failed only
`task1/seed2`.

---

# WebVLA M1 — hosted queue-mode demo

M1 puts the validated M0 closed loop behind a web demo: a visitor picks a
LIBERO task (optionally with their own instruction phrasing), submits it, and
minutes later watches the recorded episode mp4 with success/failure + stats.
One job runs at a time; others queue visibly. The M0 CLI scripts are untouched
and still work.

## Live demo

- Space: https://huggingface.co/spaces/swn21/webvla-libero-octo
  (Docker SDK, cpu-basic, requires HF PRO on the owner's account)
- Checkpoint: https://huggingface.co/swn21/octo-libero-spatial-ft (the M0
  fine-tuned 50k-step model, uploaded by `deploy/push_ckpt.py`)

![Space UI](assets/space_screenshot.png)

Verified on the live Space (2026-10-12, `deploy/verify_space.py` +
`deploy/test_space_failure.py` + human browser pass):

- `/healthz` ok (cpu backend, warmup done), 10 LIBERO-Spatial tasks listed.
- Full episode via API: task 0 / seed 0 → done, 220 steps, 135 s wall,
  replan latency mean 1367 ms (CPU), mp4 served as `video/mp4`.
- Human pass (browser): task submitted from the UI → **SUCCESS**, 168 steps,
  137 s wall, mp4 played in browser (screenshot above).
- Induced failure (invalid task_id) → `status=failed` with traceback;
  server and worker stayed alive.
- Note: CPU+OSMesa trajectories diverge numerically from the local GPU+EGL
  runs (closed-loop chaos), so Space outcomes don't reproduce local success
  rates seed-for-seed — the same seed can succeed locally and fail there.

## Architecture

```
browser (Gradio UI, mounted at "/")
   |  POST /episode {task_id, seed, instruction?}
   v
FastAPI (server/app.py)  ---- GET /episode/{job_id} <---- gr.Timer poll (2s)
   |                        GET /episodes (gallery)
   |  asyncio.Queue (the ONLY queue; Gradio's own queue is unused)
   v
Worker (server/worker.py): env + Octo policy stay loaded between jobs
   |  closed loop: obs -> action chunk -> execute K steps -> replan
   v
episodes/<job>.mp4 + <job>.json  (served back via /episodes static mount)
```

- `server/jobs.py` — in-memory JobStore; a crashed episode becomes
  `status=failed` with its traceback, it never kills the worker or the server.
- `web/ui.py` — Gradio Blocks: task dropdown (+ scene preview images from
  `assets/scenes/`), seed, optional custom instruction, status polling,
  result video with SUCCESS / FAILED / UNCHECKED-custom badge, gallery of all
  past episodes read from `episodes/`.
- `server/custom_env.py` — CustomInstructionWrapper for custom phrasings
  (success checker still belongs to the scene's original task → UNCHECKED).

## Run locally

```bash
bash setup/run.sh -m uvicorn server.app:app --host 0.0.0.0 --port 7860
# then open http://127.0.0.1:7860/  (API at /episode, /episodes, /healthz)
```

GPU by default locally; `WEBVLA_BACKEND=cpu` forces CPU inference
(`JAX_PLATFORMS=cpu`; measured 1.23 s/replan on 2 pinned cores, so episodes
take a few minutes on CPU).

## Deploy to HF Spaces

```bash
HF_TOKEN=... HF_HUB_OFFLINE=0 bash setup/run.sh deploy/push_ckpt.py   # once
HF_TOKEN=... HF_HUB_OFFLINE=0 bash setup/run.sh deploy/push_space.py  # every update
```

`push_space.py` stages the runtime subset (`server/ web/ runner/ sim/
policy/ assets/ app.py paths.py`, plus `requirements-space.txt` →
`requirements.txt` and `deploy/Dockerfile.space` → `Dockerfile`) and pushes
it as a **Docker SDK** Space on cpu-basic (HF builds the image remotely; no
local Docker needed).
Notes:

- Docker SDK was forced on us by two Gradio-SDK build failures:
  (a) the runtime force-installs the latest gradio (6.30), which needs
  `huggingface-hub>=1.16`, while transformers 4.34.1 needs `<1.0` — pip
  cannot resolve them together; (b) pip-installing LIBERO from its codeload
  tarball drops all data files (`.bddl`/`.init`/assets — no MANIFEST.in),
  verified locally. The Dockerfile installs the core stack with the hub pin,
  adds gradio `--no-deps`, and puts the pinned LIBERO source tree on
  `sys.path` via `.pth` with a pre-written `~/.libero/config.yaml`.
- The Space image pre-pulls the fine-tuned checkpoint
  (`hf://swn21/octo-libero-spatial-ft`, 810MB) and the T5 tokenizer at build
  time, so the first visitor doesn't wait on downloads.
- Rendering on the Space is OSMesa (`libosmesa6` apt-installed in the image).
- ZeroGPU was evaluated and dropped: it is Gradio-SDK + PyTorch-only, and the
  pinned `jaxlib 0.4.20+cudnn89` predates ZeroGPU's Blackwell (sm_120)
  hardware, so CPU inference on cpu-basic is the supported path.
- Local Docker was evaluated and dropped for development; the Dockerfile here
  is deploy-only and is built by HF, never locally.
- Deploy requires an HF PRO subscription (HF rejects Gradio/Docker Space
  creation on cpu-basic otherwise — `402 Payment Required`).

## Known limitations (M1)

- Episodes are slow on the Space: CPU-only inference (~1.2 s/replan) means a
  full episode takes minutes, not seconds.
- Spaces filesystem is ephemeral: the gallery shows the current container's
  episodes only; a restart wipes them. No persistence hacks by design.
- Custom instructions are UNCHECKED and often fail: the policy was fine-tuned
  on only the 10 LIBERO-Spatial instructions, so novel phrasings usually
  degrade to the scene's default behavior. That gap is part of the demo.
- One job at a time, queue is in-memory: a Space restart drops queued jobs.

---

# WebVLA M2 — live streaming + mid-episode instruction injection

M2 adds two things on top of the M1 demo, without touching the M0/M1 behavior:

- **Live view**: episodes stream frames over WebSocket while they run. The
  worker (executor thread) JPEG-encodes every frame into a per-job
  `Broadcaster` (`server/stream.py`, bounded drop-oldest subscriber queues —
  a slow viewer can't back-pressure the rollout); `WS /ws/jobs/{job_id}`
  streams them, and a small same-origin page (`server/static/live.html`,
  served at `/live?job_id=...`) renders them in an iframe inside the Gradio
  UI (Gradio strips inline scripts — hence the iframe). A polling fallback
  exists at `GET /episode/{job_id}/frame.jpg`.
- **Instruction injection**: `POST /episode/{job_id}/instruction` queues a new
  instruction for a running job; the runner picks it up at the next replan
  boundary and calls `policy.set_instruction()` (swaps the language
  conditioning, keeps the history window). Each injection is recorded in
  metadata `instruction_events` and the episode badge shows UNCHECKED with
  the change log — the sim success checker still belongs to the original
  task.

Runner changes are additive-only: `run_episode_frames(..., hooks=None)` is
byte-identical M0 behavior when no hooks are passed (regression-tested by
`setup/test_m0_regression.py`).

Verified locally (`setup/test_stream.py`): 221 live frames @ 256x256,
injection picked up at step 12, 409 on post-completion injection, metadata +
mp4 intact. Verified on the Space (`deploy/verify_space.py`): 221 frames over
wss through the HF proxy, episode + mp4 complete.

# WebVLA live mode — interactive episodes (click-to-push)

A NEW interactive layer alongside the validated pipeline (nothing in
`sim/`, `policy/`, `runner/`, `server/` is modified — see `INTEGRATION.md`
for the seam analysis). One episode runs continuously in a long-lived
session, streams to the browser over WebSocket, and the viewer can click
the live video to shove objects while the policy reacts.

## Run

```bash
# inside WSL, from the repo root (separate process from the M1/M2 server)
bash setup/run.sh -m uvicorn live.app:app --host 0.0.0.0 --port 7870
# then open http://localhost:7870/
```

Pick a task, Start, then click the video: the server unprojects the click
through the agentview camera (intrinsics + extrinsics read from the sim at
runtime), intersects the table plane, and applies a velocity impulse to the
nearest object within 12 cm — a shove, not a teleport. Debug dots (on by
default) show where the server thinks each click landed: green = object
pushed, red = nothing in range, blue = off the table.

## How it works

- `live/session.py` — `LiveSession` thread owns its own
  `LiberoWrapper(render_size=640)` + `OctoPolicy` (same checkpoint and obs
  pipeline as M0; the policy input is resized to 256^2/128^2 exactly as
  before). Loop: obs -> policy -> (4,7) chunk -> 20 Hz-paced steps ->
  replan. Start/reset/stop are commands into the thread; clicks are
  serialized with sim steps via a lock (qvel writes land between steps).
- Stream: agentview render (640x640) -> rotate 180 (the canonical upright
  view) -> center-crop 640x480 -> JPEG -> `FrameHub` (newest-frame-only:
  one shared slot, slow clients skip frames, never a backlog).
- `live/app.py` — FastAPI: `GET /` (page), `GET /tasks`, `WS /ws`,
  `GET /session`, `POST /session/start|reset|stop`, `POST /click`.
- `live/index.html` — ~150 lines of vanilla JS, no build step.

## Verified

`live/_selftest.py` (in-process, ground truth): every movable object's
projected pixel -> ray-table hit within 8 mm of its body, reprojection
round-trip 0.0 px, impulse moves the pushed object and it stays on the
table. `live/_server_check.py` (against the running server): 10 tasks
listed, start 200 / double-start 409, 51 frames in 12.2 s with 0 dropped
and <1 ms mean frame age, click round-trips 0.0 px at 4 points, push
selects the right object, reset restarts with the new seed, stop returns
to idle, policy reached success mid-test.

## Known limitation: frame rate on this machine

The spec targets ~10 fps; this machine delivers ~4.3 new frames/s. The cap
is sim step cost, not the stream: `env.step` is ~220 ms here because WSL2
has no GPU GL (only Mesa EGL — llvmpipe; OSMesa is similarly CPU-bound),
and the cost is flat vs resolution (identical at 256^2 and 640^2 —
render-pipeline fixed overhead, not pixels; see `live/_profile.py`,
`live/_gltiming.py`). M0's batch runs had the same step rate. The pacing
loop already targets real-time 20 Hz, so on a native-Linux GPU box with
real EGL the same code streams 15-20 fps. Physics is real-time-paced, so
shoves dissipate naturally.

# WebVLA π0.5 migration — Octo → openpi π0.5 (2026-10-13)

The policy layer is now swappable. **π0.5 (`pi05_libero`, Physical
Intelligence's official LIBERO fine-tune) is integrated as an alternative
backend; Octo remains the default and the M0-validated baseline.** Decision,
motivation and validation discipline: `MIGRATION.md`.

How it works: π0.5 runs in openpi's official policy server (separate venv —
openpi needs jax 0.5.3, Octo pins jax 0.4.20) and this repo talks to it over
the official websocket protocol (`policy/pi0_policy.py`, mirroring openpi's
`examples/libero/main.py`: 224×224 `resize_with_pad` images, 8-D state,
10-action chunks, 5 executed per replan). The sim, runner, queue server and
live mode are untouched; each picks the backend via one flag.

## Results (LIBERO-Spatial, this machine)

| eval | success | notes |
|---|---|---|
| π0.5 reference (official openpi client, 10 tasks × 3 trials) | **27/30 = 90.0%** | published π0.5 LIBERO avg: 98.8%; log: `results/pi05_reference_eval.log` |
| π0.5 integrated (`run_batch.py --policy pi05`, tasks 0-2 × seeds 0-4) | **15/15 = 100%** | `results_pi05.json`; replan latency mean 323 ms |
| Octo fine-tuned baseline (same grid, M0) | 11/15 = 73.3% | `results.json`; replan latency mean 157 ms |

Sanity check (obs keys vs policy-expected keys): MATCH for both backends,
recorded in each results file.

## Run with π0.5

```bash
# 1) policy server, once (~5 min load, ~13 GB RAM, wants the whole GPU):
wsl bash setup/run_pi05_server.sh            # logs: listening on 0.0.0.0:8000

# 2) batch / server / live — Octo stays the default everywhere:
wsl bash setup/run.sh run_batch.py --policy pi05 --results results_pi05.json
WEBVLA_POLICY=pi05 wsl bash setup/run.sh -m uvicorn server.app:app --port 7860
WEBVLA_POLICY=pi05 wsl bash setup/run.sh -m uvicorn live.app:app --port 7870
```

`WEBVLA_PI05_SERVER=host:port` overrides the default `127.0.0.1:8000`.

## Operational notes (hard-won, details in MIGRATION.md)

- Start the π0.5 server **before** other GPU processes — JAX preallocates
  ~75% of VRAM and a starved server runs 20× slower.
- The stock openpi websocket client's 20 s keepalive ping kills connections
  during the ~20-40 s first-inference XLA compile; `Pi05Policy` disables
  pings and warms up at init. If you drive the server with the *official*
  eval client, warm it first (`wsl bash setup/test_pi05_infer.sh`).
- `setup/run.sh` sets `TF_FORCE_GPU_ALLOW_GROWTH=true` — without it TF
  reserves ~9 GB at import and starves the policy server.
