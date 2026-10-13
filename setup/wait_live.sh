#!/bin/bash
for i in $(seq 1 60); do
  if curl -s -m 2 http://127.0.0.1:7870/tasks 2>/dev/null | grep -q '"ready": *true'; then
    echo READY
    exit 0
  fi
  sleep 3
done
echo TIMEOUT
exit 1
