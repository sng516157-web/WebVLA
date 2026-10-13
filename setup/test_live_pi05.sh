#!/usr/bin/env bash
# Exercise the live-mode HTTP API end to end against a running server (7870).
set -euo pipefail
echo "== status before:"; curl -s --max-time 5 http://127.0.0.1:7870/session; echo
echo "== start:"; curl -s --max-time 10 -X POST http://127.0.0.1:7870/session/start \
    -H 'Content-Type: application/json' -d '{"task_id": 0, "seed": 0}'; echo
for i in 1 2 3 4 5 6; do
    sleep 10
    echo "== status +${i}0s:"; curl -s --max-time 5 http://127.0.0.1:7870/session; echo
done
echo "== stop:"; curl -s --max-time 10 -X POST http://127.0.0.1:7870/session/stop; echo
sleep 2
echo "== status after stop:"; curl -s --max-time 5 http://127.0.0.1:7870/session; echo
