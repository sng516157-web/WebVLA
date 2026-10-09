#!/bin/bash
source /root/webvla-env.sh
for p in \
  "examples/06_eval_finetuned_on_libero.py" \
  "examples/05_eval_finetuned_on_libero.py" \
  "examples/04_eval_finetuned_on_libero.py" \
  "examples/eval_libero.py" \
  "scripts/eval_libero.py" \
  "scripts/evaluate_libero.py" \
  "scripts/evaluate.py" ; do
  n=$(timeout 30 curl -s "https://api.github.com/repos/octo-models/octo/commits?path=$p" | grep -c '"sha"')
  echo "$p -> $n"
done
# also list all tags
echo "=== tags ==="
timeout 30 curl -s "https://api.github.com/repos/octo-models/octo/tags" | grep '"name"'
