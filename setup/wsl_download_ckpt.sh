#!/bin/bash
WEBVLA_HOME=${WEBVLA_HOME:-/root/webvla}
WEBVLA_ENV_FILE=${WEBVLA_ENV_FILE:-$HOME/webvla-env.sh}
[ -f "$WEBVLA_ENV_FILE" ] && source "$WEBVLA_ENV_FILE" || true
PY=$WEBVLA_HOME/venv/bin/python
$PY - <<'EOF'
import huggingface_hub
p = huggingface_hub.snapshot_download("rail-berkeley/octo-base-1.5")
print("CHECKPOINT_AT:", p)
EOF
echo "CKPT_DOWNLOAD_DONE"
