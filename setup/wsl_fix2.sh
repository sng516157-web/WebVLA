#!/bin/bash
set -euo pipefail
WEBVLA_HOME=${WEBVLA_HOME:-/root/webvla}
WEBVLA_ENV_FILE=${WEBVLA_ENV_FILE:-$HOME/webvla-env.sh}
[ -f "$WEBVLA_ENV_FILE" ] && source "$WEBVLA_ENV_FILE" || true
source /root/.local/bin/env
PY=$WEBVLA_HOME/venv/bin/python

uv pip install --python $PY "tensorflow-datasets==4.9.2" "tensorflow-metadata==1.14.0"
uv pip install --python $PY "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu

$PY - <<'EOF'
import torch, tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import dlimp
from libero.libero import benchmark
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
init = suite.get_task_init_states(0)
print("torch", torch.__version__, "| init states:", len(init))
print("IMPORTS_OK")
EOF
echo "FIX2_DONE"
