#!/bin/bash
PY=/root/webvla/venv/bin/python
$PY - <<'EOF'
import traceback
for mod in ["jaxlib.cuda_plugin_extension", "jaxlib.cuda._prng", "jaxlib.gpu_prng", "jaxlib.gpu_common_utils"]:
    try:
        __import__(mod)
        print("OK   ", mod)
    except Exception as e:
        print("FAIL ", mod, "->", repr(e)[:200])
import jaxlib.gpu_common_utils as gcu
print([a for a in dir(gcu) if not a.startswith("__")])
EOF
