#!/bin/bash
# Wait for the local WebVLA server to answer /healthz (up to 5 min).
for i in $(seq 1 30); do
  h=$(curl -s http://127.0.0.1:7860/healthz 2>/dev/null)
  if [ -n "$h" ]; then
    echo "$h"
    exit 0
  fi
  sleep 10
done
echo TIMEOUT
tail -20 /tmp/webvla_server.log
exit 1
