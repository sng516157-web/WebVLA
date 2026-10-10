#!/bin/bash
# Wait for warmup, then check the UI page + /tasks endpoint (run inside WSL).
for i in $(seq 1 40); do
  out=$(curl -s --max-time 3 http://127.0.0.1:7860/healthz 2>/dev/null)
  case "$out" in *'"warmup_done":true'*) echo "WARM: $out"; break;; esac
  sleep 5
done
echo "--- GET / :"
curl -s --max-time 5 -o /tmp/page.html -w "http=%{http_code} bytes=%{size_download}\n" http://127.0.0.1:7860/
grep -o "<title>[^<]*</title>" /tmp/page.html | head -1
echo "--- GET /tasks :"
curl -s --max-time 5 http://127.0.0.1:7860/tasks | head -c 400; echo
