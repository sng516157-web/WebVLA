#!/bin/bash
# Start the local WebVLA server detached (logs to /tmp/webvla_server.log).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
nohup bash setup/run.sh -m uvicorn server.app:app --host 0.0.0.0 --port 7860 \
  > /tmp/webvla_server.log 2>&1 &
echo "server pid $!"
