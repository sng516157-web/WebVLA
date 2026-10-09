#!/bin/bash
SP=/root/webvla/venv/lib/python3.10/site-packages
F="$SP/jaxlib/xla_extension.so"
echo "=== GNU_STACK before ==="
readelf -lW "$F" | grep -i stack || echo "(no GNU_STACK segment)"
echo "=== patchelf ==="
patchelf --clear-execstack "$F"; echo "patchelf exit=$?"
echo "=== GNU_STACK after ==="
readelf -lW "$F" | grep -i stack || echo "(no GNU_STACK segment)"
echo "=== import test ==="
/root/webvla/venv/bin/python -c "import jax; print('jax ok:', jax.__version__); print(jax.devices())"
