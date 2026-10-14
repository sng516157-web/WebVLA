#!/usr/bin/env bash
# Report whether the local demo ports are serving anything.
for p in 8000 7860 7870; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://127.0.0.1:${p}/")
  echo "port ${p}: ${code:-no-response}"
done
ps aux | grep -E "uvicorn|run_pi05|policy_server" | grep -v grep || echo "no matching processes"
