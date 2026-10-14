#!/usr/bin/env bash
# Spot-check the hookability claims in the installed LeRobot pi05 model.
M=/root/webvla/venv_lerobot/lib/python3.10/site-packages/lerobot/policies/pi05/modeling_pi05.py
echo "== eager attention forced at inference =="
grep -n '_attn_implementation = "eager"' "$M"
echo "== denoise loop =="
grep -n 'for step in range(num_steps)' "$M"
grep -n 'x_t = x_t + dt' "$M"
echo "== key methods =="
grep -n 'def sample_actions\|def denoise_step\|def embed_prefix\|def predict_action_chunk' "$M"
echo "== empty camera padding =="
grep -n 'empty_cam' "$M" | head -5
echo "== output_attentions support in fork gemma =="
G=$(python3 - <<'EOF'
import glob
hits = glob.glob('/root/webvla/venv_lerobot/lib/python3.10/site-packages/transformers/models/gemma/modeling_gemma.py')
print(hits[0] if hits else '')
EOF
)
grep -n 'output_attentions' "$G" | head -5
