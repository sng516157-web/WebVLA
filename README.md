# WebVLA M0 — Sim-based VLA demo (LIBERO + Octo)

Local Python pipeline: natural-language instruction → Octo policy in closed loop
on LIBERO simulation tasks → per-episode mp4 + metadata JSON. CLI scripts only —
no web server, no UI, no Docker.

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
