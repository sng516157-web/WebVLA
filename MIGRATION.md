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
4. **Faster episodes.** π0 emits 50-step action chunks vs Octo's 4 — replans
   drop ~12×. This shrinks the CPU-inference penalty on the HF Space and
   shortens GPU episodes everywhere.
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
