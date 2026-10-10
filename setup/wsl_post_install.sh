#!/bin/bash
set -euo pipefail
WEBVLA_HOME=${WEBVLA_HOME:-/root/webvla}
WEBVLA_ENV_FILE=${WEBVLA_ENV_FILE:-$HOME/webvla-env.sh}
[ -f "$WEBVLA_ENV_FILE" ] && source "$WEBVLA_ENV_FILE" || true
source /root/.local/bin/env
PY=$WEBVLA_HOME/venv/bin/python

# torch CPU: required at runtime — LIBERO get_task_init_states() uses torch.load.
# matplotlib: imported by libero env_wrapper (matplotlib.cm). >=3.6.1 required for
# numpy 1.24 compat (documented deviation from LIBERO's 3.5.3 pin).
uv pip install --python $PY matplotlib==3.7.5
uv pip install --python $PY torch --index-url https://download.pytorch.org/whl/cpu

# Pre-create LIBERO config non-interactively (first import otherwise prompts).
mkdir -p /root/.libero
cat > /root/.libero/config.yaml <<EOF
benchmark_root: $WEBVLA_HOME/LIBERO/libero/libero
bddl_files: $WEBVLA_HOME/LIBERO/libero/libero/bddl_files
init_states: $WEBVLA_HOME/LIBERO/libero/libero/init_files
datasets: $WEBVLA_HOME/LIBERO/libero/datasets
assets: $WEBVLA_HOME/LIBERO/libero/libero/assets
EOF

echo "POST_INSTALL_DONE"
