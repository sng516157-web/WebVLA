# WebVLA M0 — Sim-based VLA demo (LIBERO + Octo)

Local Python pipeline: natural-language instruction → Octo policy in closed loop
on LIBERO simulation tasks → per-episode mp4 + metadata.

Status: scaffolding. Environment details and reference numbers will be recorded
here as they are produced (nothing is reported that does not come from a file on
disk).

## Layout

- `sim/wrapper.py` — thin LIBERO env wrapper (reset/step/render/obs)
- `runner/run_episode.py` — closed-loop episode generator → mp4 + metadata JSON
- `run_batch.py` — batch runner → `episodes/` + `results.json`

## Environment

Runs inside WSL2 (Ubuntu) on this Windows machine; native Windows is not viable
(JAX CUDA wheels are Linux-only). Third-party code lives in WSL at
`/root/webvla/{LIBERO,octo}`; this repo is mounted at
`/mnt/c/Users/sng51/Desktop/WebVLA`.
