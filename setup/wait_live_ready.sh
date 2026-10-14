#!/usr/bin/env bash
# Wait until the live-mode server reports tasks ready.
for i in $(seq 1 90); do
  body=$(curl -s --max-time 2 http://127.0.0.1:7870/tasks || true)
  case "$body" in
    *'"ready":true'*) echo "READY after ~$((i*5))s"; exit 0;;
  esac
  sleep 5
done
echo "server never became ready"; exit 1
