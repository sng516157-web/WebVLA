# MIGRATION.md — Octo → π0.5 (openpi)

## What

Migrate the policy layer of WebVLA from **Octo (octo-base-1.5, fine-tuned
in-house on LIBERO-Spatial)** to **π0.5 (`pi05_libero`) via Physical
Intelligence's openpi**, keeping the rest of the project — LIBERO sim stack,
episode runner, queue server, live mode — intact.

The policy is the only swapped component. Everything talks to it through one
interface (`reset(instruction)`, `act(obs) -> action chunk`), so the change
is: a new `policy/` wrapper, π0.5's obs conventions (224×224 images, 8-D
proprio, its own action de-normalization via openpi's official LIBERO eval
recipe), and a fresh reference validation before anything builds on it.

## When

Started 2026-10-13, after M0 (validated Octo pipeline), M1 (queue demo on HF
Spaces), M2 (live streaming + instruction injection), and live mode
(interactive click-to-push) were all complete and pushed.

## Why

1. **Modern, maintained stack.** Octo pins jax 0.4.20 / flax 0.7.5 / TF 2.15
   (Sept–Dec 2023). That stack predates Blackwell GPUs, blocks ZeroGPU (which
   now runs RTX Pro 6000 Blackwell), and forces the transformers/gradio
   dependency triangle that made the Space Docker-only. openpi is actively
   developed; PyTorch support landed Sept 2025 and is validated on LIBERO.
2. **No retraining needed.** Physical Intelligence publishes `pi05_libero`,
   an official LIBERO fine-tuned π0.5 checkpoint (SOTA on the benchmark).
   We adopt it the same way we adopted octo-base: plug in, validate against
   published numbers, then build.
3. **Better expected behavior.** π0.5's LIBERO results are well above Octo's
   (both our measured 73–83% and Octo's published numbers), and its much
   larger pretraining mixture should improve the custom-instruction
   generalization that was visibly weak with our Octo fine-tune.
4. **Faster episodes.** π0.5 emits 10-step action chunks (executing 5 per
   replan, per the official recipe) vs Octo's 4-step horizon, and inference
   is ~0.3s on the GPU — replan count and latency both drop. This shrinks
   the CPU-inference penalty on the HF Space and shortens GPU episodes
   everywhere.
5. **Deployment options.** A PyTorch policy stack is ZeroGPU-compatible
   (ZeroGPU's runtime is PyTorch-native) and removes the Docker-SDK
   constraint, while T4-class persistent GPUs work regardless.

## What stays

- LIBERO sim stack (mujoco 2.3.2 / robosuite 1.4.x pins) — independent of
  the policy.
- `runner/`, `server/`, `live/`, `run_batch.py` — unchanged consumers of the
  policy interface.
- **Octo remains available**: `policy/octo_policy.py`, the fine-tuned
  checkpoint, and all M0/M1/M2 artifacts stay in the repo. A `--policy`
  selector (`octo` | `pi05`) picks the backend; Octo results and the
  M0 validation record stay documented in the README as the original
  baseline.

## Validation discipline (same as M0)

Step 1 before any integration: run openpi's official LIBERO evaluation with
`pi05_libero` on 2 tasks × 3 episodes, record the success rate, compare to
published π0.5 LIBERO numbers. If it deviates badly, stop and report the
suspected cause — no workaround code on an unvalidated stack.

## Validation results (2026-10-13)

- **Reference eval (official openpi client, `examples/libero/main.py`,
  `pi05_libero` checkpoint, LIBERO-Spatial): 27/30 = 90.0%** (10 tasks ×
  3 trials; raw log `results/pi05_reference_eval.log`). Published π0.5
  LIBERO average is 98.8%. Per-task: 8 of 10 tasks went 3/3; task 3 ("bowl
  on cookie box") 1/3, task 9 ("bowl on wooden cabinet") 2/3. At n=3 per
  task the confidence interval is wide, the stack integrity checks were all
  clean, and the two weak tasks are known-hard — verdict: not a bad
  deviation, proceed.
- **Extended reference eval (official client, 10 tasks × 10 trials):
  97/100 = 97.0%** (raw log `results/pi05_reference_eval_10trials.log`).
  Seven tasks 10/10, three tasks 9/10 — within a point of the published
  98.8% LIBERO-Spatial number. The first attempt at this run silently
  zeroed all 100 episodes (websocket keepalive death on a GPU-starved
  server — see Operational findings); the number above is the healthy
  rerun.
- **Integration smoke test** (`run_batch.py --policy pi05`, task 0 seed 0):
  SUCCESS in 74 steps; sanity check MATCH (`observation/image`,
  `observation/wrist_image`, `observation/state` shapes as expected);
  replan latency mean 309 ms on the RTX 4080 Laptop GPU.
- **Octo regression**: `setup/test_m0_regression.py` PASSes — the Octo code
  path is behavior-identical after the integration edits.

## Operational findings (read before running anything)

1. **The π0.5 policy server must own the GPU.** JAX preallocates ~75% of
   VRAM per process; with the two Octo demo servers also on the GPU the
   π0.5 server was squeezed to host-memory speed (4-7 s/inference instead
   of 0.3 s). Start it first (or alone): `setup/run_pi05_server.sh`.
2. **Disable the websocket keepalive ping timeout.** openpi's stock client
   uses websockets' 20 s ping default; the first inference per input
   signature triggers an XLA compile of ~20-40 s (single-threaded server
   can't answer pings meanwhile) and the client kills the connection
   mid-compile — this silently zeroed an entire 100-episode eval run
   (every episode failed fast, recorded as Failure). `Pi05Policy` patches
   `ping_interval=None` and warms up at init with obs-identical dtypes.
   When using the *official* eval client, warm the server first
   (`setup/test_pi05_infer.sh`) and note its state dtype is float64, which
   is a different XLA signature than a float32 warmup.
3. **TF grabs GPU memory even when unused.** `sim/wrapper.py` imports
   TensorFlow for image resizing (CPU ops); at init TF reserved ~9 GB and
   starved the policy server (7.5 s/replan). `setup/run.sh` now sets
   `TF_FORCE_GPU_ALLOW_GROWTH=true`. Replan latency dropped 7576 → 309 ms.
4. **RAM:** the policy server needs ~13 GB; WSL is configured for 26 GB
   (`.wslconfig`) plus a 16 GB swapfile (`swapon /swapfile_pi05` after any
   `wsl --shutdown`).
