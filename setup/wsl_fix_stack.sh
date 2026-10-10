#!/bin/bash
set -euo pipefail
WEBVLA_HOME=${WEBVLA_HOME:-/root/webvla}
WEBVLA_ENV_FILE=${WEBVLA_ENV_FILE:-$HOME/webvla-env.sh}
[ -f "$WEBVLA_ENV_FILE" ] && source "$WEBVLA_ENV_FILE" || true
source /root/.local/bin/env
PY=$WEBVLA_HOME/venv/bin/python
SP=$WEBVLA_HOME/venv/lib/python3.10/site-packages

# Fix broken editable mapping for libero (uv/setuptools PEP660 produced empty MAPPING).
echo "$WEBVLA_HOME/LIBERO" > "$SP/libero-repo.pth"

# cudnn 8.9 required by jaxlib 0.4.20+cuda12.cudnn89 (cudnn 9 has different soname).
uv pip install --python $PY "nvidia-cudnn-cu12==8.9.2.26"

# Clean reinstall of exact jax/jaxlib wheels (suspected mixed-version files).
uv pip install --python $PY --no-cache --force-reinstall \
  "jax==0.4.20" "jaxlib==0.4.20+cuda12.cudnn89" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "=== verify ==="
$PY -c "import jax; print('jax', jax.__version__, 'jaxlib', jax.lib and 'ok'); print(jax.devices())"
$PY -c "import libero.libero; print('libero import ok')"
echo "FIX_DONE"
