#!/usr/bin/env bash
# Set HF_TOKEN as a Space secret so the Space can make authenticated HF Hub
# calls at boot (faster model downloads; gated-tokenizer fallback).
# Usage: HF_TOKEN=hf_... bash setup/set_space_secret.sh
set -e
: "${HF_TOKEN:?set HF_TOKEN in the environment}"
source ~/webvla-env.sh
cd /mnt/c/Users/sng51/Desktop/WebVLA
cat > /tmp/set_space_secret.py <<'EOF'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.add_space_secret("swn21/webvla-libero-pi05", "HF_TOKEN", os.environ["HF_TOKEN"])
print("HF_TOKEN secret set on swn21/webvla-libero-pi05")
EOF
HF_HUB_OFFLINE=0 bash setup/run.sh /tmp/set_space_secret.py
