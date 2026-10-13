#!/bin/bash
# Wait for port 8000 to listen, then show server log tail + memory.
for i in $(seq 1 54); do
  if ss -tln 2>/dev/null | grep -q :8000; then
    echo "LISTENING after ~$((i*10))s"
    break
  fi
  sleep 10
done
tail -4 /root/webvla/openpi_server.log
free -g | head -3
