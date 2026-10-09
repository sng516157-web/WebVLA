#!/bin/bash
source /root/webvla-env.sh
PY=/root/webvla/venv/bin/python
$PY - <<'EOF'
import huggingface_hub as hf
files = hf.list_repo_files("yifengzhu-hf/LIBERO-datasets", repo_type="dataset")
for f in files:
    print(f)
EOF
