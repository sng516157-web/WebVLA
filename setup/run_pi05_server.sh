#!/usr/bin/env bash
# Start the pi0.5 policy server (openpi official serve_policy, LIBERO env).
# Runs OUTSIDE the project venv, in openpi's own uv environment, because
# openpi needs jax 0.5.3 which cannot coexist with Octo's pinned jax 0.4.20.
# The model takes ~4.5 min to load; port 8000. See setup/wait_8000.sh.
set -euo pipefail
source ~/webvla-env.sh 2>/dev/null || true
cd /root/webvla/openpi
exec /root/.local/bin/uv run scripts/serve_policy.py --env LIBERO policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero
