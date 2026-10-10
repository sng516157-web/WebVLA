#!/bin/bash
set -euo pipefail
WEBVLA_HOME=${WEBVLA_HOME:-/root/webvla}
WEBVLA_ENV_FILE=${WEBVLA_ENV_FILE:-$HOME/webvla-env.sh}
[ -f "$WEBVLA_ENV_FILE" ] && source "$WEBVLA_ENV_FILE" || true
source /root/.local/bin/env
PY=$WEBVLA_HOME/venv/bin/python

uv pip install --python $PY "numpy==1.24.3"

$PY - <<'EOF'
import numpy as np
print("numpy", np.__version__)
import cv2
print("cv2", cv2.__version__)
import tensorflow as tf
print("tf", tf.__version__)
tf.config.set_visible_devices([], "GPU")
import jax
print("jax", jax.__version__, jax.devices())
import robosuite
print("robosuite", robosuite.__version__)
EOF
echo "NUMPY_FIX_DONE"
