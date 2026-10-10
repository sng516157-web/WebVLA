#!/bin/bash
WEBVLA_HOME=${WEBVLA_HOME:-/root/webvla}
WEBVLA_ENV_FILE=${WEBVLA_ENV_FILE:-$HOME/webvla-env.sh}
[ -f "$WEBVLA_ENV_FILE" ] && source "$WEBVLA_ENV_FILE" || true
PY=$WEBVLA_HOME/venv/bin/python
$PY - <<'EOF'
import huggingface_hub as hf
files = hf.list_repo_files("yifengzhu-hf/LIBERO-datasets", repo_type="dataset")
for f in files:
    print(f)
EOF
