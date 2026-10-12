---
title: WebVLA — LIBERO + Octo queue demo
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# WebVLA — queue-mode VLA demo

Pick a LIBERO-Spatial task, submit it, and minutes later watch the recorded
episode (mp4) with success/failure and inference-latency stats. One job runs
at a time; the rest queue visibly.

Policy: Octo (octo-base-1.5) fine-tuned on LIBERO-Spatial (50k steps),
running in closed loop in LIBERO's MuJoCo sim. Source + validation numbers:
https://github.com/sng51/WebVLA (M0: reference eval 5/6 = 83.3% vs published
~78.9%; batch 11/15 = 73.3%).

Notes:
- Episodes render offscreen via OSMesa (CPU software rendering, no display).
- The gallery shows episodes from the current container only — the Spaces
  filesystem is ephemeral.
- Inference runs on CPU (pinned jaxlib 0.4.20 predates ZeroGPU's Blackwell
  hardware and ZeroGPU is PyTorch-only); episodes take a few minutes.
