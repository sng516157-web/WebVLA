#!/bin/bash
source /root/webvla-env.sh
PY=/root/webvla/venv/bin/python
$PY - <<'EOF'
import huggingface_hub
p = huggingface_hub.snapshot_download("rail-berkeley/octo-base-1.5")
print("CHECKPOINT_AT:", p)
EOF
echo "CKPT_DOWNLOAD_DONE"
