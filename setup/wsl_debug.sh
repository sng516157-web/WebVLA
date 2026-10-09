#!/bin/bash
PY=/root/webvla/venv/bin/python
$PY -c "import tensorflow" 2>&1 | tail -25
echo "==="
source /root/.local/bin/env
uv pip list --python $PY | grep -i -E "ml.dtypes|h5py|scipy|opt-einsum|numpy|tensorboard"
