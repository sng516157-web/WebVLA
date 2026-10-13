# π0.5 vs Octo on LIBERO-Spatial — performance report

Date: 2026-10-13. All numbers are read from files in this repo
(`results.json`, `results_pi05.json`, `results_lerobot_pi05.json`,
`results/pi05_reference_eval*.log`, `reference/**/reference_results.json`,
per-episode metadata in `episodes/`), never from memory.

## 1. Success rate

Same closed-loop harness, same tasks, same seeds (LIBERO-Spatial tasks 0–2 ×
seeds 0–4, 220-step cap):

| policy | task 0 | task 1 | task 2 | overall |
|---|---|---|---|---|
| Octo fine-tuned (50k steps, this repo) | 3/5 | 4/5 | 4/5 | **11/15 = 73.3%** |
| π0.5 (JAX policy server) | 5/5 | 5/5 | 5/5 | **15/15 = 100%** |
| π0.5 (LeRobot PyTorch port, 10 tasks × 3 seeds) | — | — | — | **30/30 = 100%** |

Against published numbers:

- Octo fine-tuned: published 78.9% on LIBERO-Spatial (OpenVLA paper Table 3).
  Our M0 reference eval scored 5/6 = 83.3% (2 tasks × 3 seeds) and the batch
  grid 73.3% — consistent.
- π0.5: official openpi eval client scored 27/30 = 90.0%, extended run
  **97/100 = 97.0%** (10 tasks × 10 trials) vs the published 98.8% — the
  stack reproduces the published result.

## 2. Failure modes

- **Octo:** all 4 failures ran to the 220-step cap (`task0/seed1`,
  `task0/seed2`, `task1/seed4`, `task2/seed4`). Video inspection of
  `task0/seed1`: the arm reaches and grasps the correct bowl, then fumbles it
  mid-transport — grasp-physics flakiness on an otherwise correct policy, not
  scene misunderstanding.
- **π0.5:** zero failures across 45 integrated episodes. Reference-eval misses
  cluster on the two known-hard tasks (bowl-on-cookie-box, bowl-on-cabinet).

## 3. Speed (RTX 4080 Laptop, per replan; first-call warmup excluded)

| policy | replan latency mean / p50 | exec horizon | amortized / step | episode wall (successes) |
|---|---|---|---|---|
| Octo | 157 ms / 156 ms | 4 | ~39 ms | 32.7 s |
| π0.5 JAX | 313 ms / 296 ms | 5 | ~63 ms | 34.2 s |
| π0.5 LeRobot (torch) | 561 ms / 555 ms | 5 | ~112 ms | 45.0 s |
| π0.5 on ZeroGPU Space (A10G) | 590 ms | 5 | ~118 ms | 25.7 s (83-step episode) |

Octo is ~2× faster per replan than JAX π0.5 and ~3.6× faster than the torch
port — expected at ~93M params vs ~3.3B (3B PaliGemma VLM + ~300M action
expert). Wall-clock episode time locally is dominated by CPU sim stepping
(~190–220 ms/step), so the policy-speed gap mostly washes out.

## 4. Episode efficiency (successful episodes)

| policy | mean steps | min–max | replans |
|---|---|---|---|
| Octo | 105.5 | 83–140 | 26.6 |
| π0.5 JAX | 95.4 | 73–118 | 19.5 |
| π0.5 LeRobot | 104.6 | 77–193 | 21.3 |

## 5. Language generalization (custom instructions)

- **Octo:** 4 recorded custom-instruction episodes (metadata in `episodes/`):
  2 succeeded, both near-paraphrases of the scene's trained instruction; 2
  failed at the step cap on out-of-distribution phrasings. Custom phrasings
  generally degrade to the scene's default behavior.
- **π0.5:** no systematic grid recorded; qualitatively stronger (custom
  instructions have succeeded on the live demo), unquantified.

## 6. Operational cost

| | Octo | π0.5 |
|---|---|---|
| Stack | single venv, JAX in-process | separate venv + websocket policy server (jax 0.5.3 vs Octo's jax 0.4.20 pin); LeRobot torch port for ZeroGPU |
| Memory | shares GPU politely | ~13 GB RAM; JAX preallocates ~75% VRAM — must own the GPU (starved ≈ 20× slower) |
| ZeroGPU deployable | no (JAX-only pin) | yes, via LeRobot port (live, verified e2e) |
| First-call cost | JIT spike (7.5 s max observed) | XLA compile 20–40 s per input signature; warmup required |

## Bottom line

π0.5 is the better policy by every quality metric (+27 points success on the
shared grid, tighter step counts, zero failures in 45 runs) and is the only
one deployable on ZeroGPU. Octo is 2–3.6× faster per inference and far
cheaper operationally.

Comparability caveat: Octo's integrated grid covers tasks 0–2 × 5 seeds
(15 episodes); the fairest full-suite π0.5 number is the JAX reference eval,
97/100 over all 10 tasks.
