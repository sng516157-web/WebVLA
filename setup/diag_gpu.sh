#!/usr/bin/env bash
for p in 345 373 790; do
    n=$(ls -l /proc/$p/fd 2>/dev/null | grep -c dxg)
    echo "pid $p dxg fds: $n"
done
echo ---
# what does the openpi venv see right now?
cd /root/webvla/openpi
uv run python -c "import jax; print('jax devices:', jax.devices())" 2>&1 | tail -3
