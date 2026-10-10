#!/bin/bash
WEBVLA_HOME=${WEBVLA_HOME:-/root/webvla}
# Unconditionally clear execstack on every jaxlib .so (some lack PT_GNU_STACK,
# which patchelf --print-execstack reports as empty and the selective loop missed).
set -uo pipefail
SP=$WEBVLA_HOME/venv/lib/python3.10/site-packages
find "$SP/jaxlib" -name "*.so" | while read -r f; do
  patchelf --clear-execstack "$f" 2>/dev/null && echo "patched: $f" || echo "skip: $f"
done
PY=$WEBVLA_HOME/venv/bin/python
$PY -c "
import jax
from jaxlib.cuda import _prng
import jax.numpy as jnp
k = jax.random.PRNGKey(0)
a, b = jax.random.split(k)
print('GPU split ok:', a, b)
print(jax.devices())
"
echo "PRNG_FIX_DONE"
