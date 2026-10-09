# WebVLA M0 — Sim-based VLA demo (LIBERO + Octo)

Local Python pipeline: natural-language instruction → Octo policy in closed loop
on LIBERO simulation tasks → per-episode mp4 + metadata JSON. CLI scripts only —
no web server, no UI, no Docker.

## Layout

- `sim/wrapper.py` — thin LIBERO env wrapper: `reset(task_id, seed)`, `step(action)`,
  `render(camera_name) -> HWC uint8`, `obs() -> dict` (policy-keyed images, proprio, instruction)
- `policy/octo_policy.py` — Octo checkpoint loading + windowed action-chunk prediction
- `runner/run_episode.py` — closed-loop episode generator (yields frames) + mp4/metadata saver
- `runner/batch.py` — shared grid runner, sanity check, aggregation
- `reference_eval.py` — step 1: reference validation (2 tasks × 3 episodes)
- `run_batch.py` — step 4: batch run (3 tasks × 5 seeds) → `episodes/`, `results.json`
- `setup/` — WSL2 environment bootstrap scripts

## Environment (this machine)

The stack runs inside **WSL2 Ubuntu 26.04** — native Windows is not viable
(JAX CUDA wheels are Linux-only). Third-party code lives in WSL at
`/root/webvla/{LIBERO,octo}` (pinned: LIBERO master @ 8f1084e, octo main @ 241fb35);
this repo is accessed at `/mnt/c/Users/sng51/Desktop/WebVLA`.

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
- WSL networking: Windows proxy at `127.0.0.1:19828` is reachable via WSL
  VirtioProxy localhost forwarding; `setup/run.sh` exports it.
- Headless rendering: `MUJOCO_GL=egl` verified working (OSMesa verified as fallback;
  use `MUJOCO_GL=osmesa bash setup/run.sh ...`).

## How to run

```bash
# one-time setup (inside WSL, as root)
bash setup/wsl_setup_env.sh && bash setup/wsl_post_install.sh \
  && bash setup/wsl_fix_stack.sh && bash setup/wsl_fix_execstack.sh \
  && bash setup/wsl_fix_numpy.sh && bash setup/wsl_fix2.sh

# reference validation (2 tasks x 3 episodes)
wsl -d Ubuntu -- bash /mnt/c/Users/sng51/Desktop/WebVLA/setup/run.sh reference_eval.py

# batch (3 tasks x 5 seeds)
wsl -d Ubuntu -- bash /mnt/c/Users/sng51/Desktop/WebVLA/setup/run.sh run_batch.py
```

## Results

_TODO: filled from `reference/reference_results.json` and `results.json` on disk._
