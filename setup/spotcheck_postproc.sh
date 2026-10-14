#!/usr/bin/env bash
# Inspect how the pi05 checkpoint's postprocessor unnormalizes actions, and
# the denoise_step signature — needed to route intermediate x_t through the
# OFFICIAL unnormalization rather than improvising one.
set -e
V=/root/webvla/venv_lerobot/lib/python3.10/site-packages
echo "== denoise_step signature =="
sed -n '859,900p' "$V/lerobot/policies/pi05/modeling_pi05.py"
echo "== predict_action_chunk =="
sed -n '1220,1235p' "$V/lerobot/policies/pi05/modeling_pi05.py"
echo "== unnormalizer step =="
grep -rn "class UnnormalizerProcessorStep\|class .*Unnormaliz" "$V/lerobot/processor/" | head -5
