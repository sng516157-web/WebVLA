---
title: WebVLA pi05 LIBERO
emoji: 🦾
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
---

# WebVLA — π0.5 on LIBERO-Spatial (ZeroGPU)

A queue-mode demo of a vision-language-action policy driving a simulated robot
arm. Pick a LIBERO-Spatial task, submit it, and watch the recorded episode
(mp4) with success/failure and latency stats.

- **Policy**: π0.5 (`lerobot/pi05_libero_finetuned_v044`), LeRobot's PyTorch
  port of the openpi checkpoint. Validated offline at 97/100 on LIBERO-Spatial
  against the JAX reference (see the source repo below).
- **Sim**: LIBERO benchmark (robosuite 1.4 / mujoco 2.3.2), CPU + OSMesa
  rendering.
- **GPU**: inference runs on ZeroGPU — one `@spaces.GPU` call per action chunk
  (5 sim steps). The physics sim and video encoding run on CPU.

**GPU quota** (daily, per visitor): anonymous ~2 min, free HF login ~5 min,
PRO ~40 min. An episode needs roughly 1-2 minutes of GPU. If quota is
exhausted you get a clear message, not a stack trace.

**Custom instructions**: the policy was fine-tuned on only the 10 scene
instructions, so custom phrasings usually degrade to the scene's default
behavior — that generalization gap is part of what this demo shows. Custom
runs are UNCHECKED (the success checker belongs to the original task).

**Ephemeral gallery**: episodes are written to the container's disk; the
gallery resets when the Space restarts or sleeps.

**X-Ray overlays (opt-in)**: tick boxes to draw model internals into the
episode — the predicted 50-step action path (it executes 5, then replans;
the far tail can leave the frame because those steps are never actually
visited), the live 7-D action bars, the replan timeline, the 10-step
flow-matching denoising convergence (how the trajectory was refined — not
"intent"), and where instruction tokens attend (attention weights are not
an explanation of why it acted). Observation is read-only — verified
action-identical with overlays on vs off.

Source: https://github.com/sng516157-web/WebVLA
