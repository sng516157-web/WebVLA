#!/usr/bin/env bash
# Submit one job to the queue server (7860) and poll until done/failed.
set -euo pipefail
for i in $(seq 1 24); do
    if curl -s --max-time 3 http://127.0.0.1:7860/healthz >/dev/null 2>&1; then break; fi
    sleep 5
done
echo "== submit:"
RESP=$(curl -s --max-time 10 -X POST http://127.0.0.1:7860/episode \
    -H 'Content-Type: application/json' -d '{"task_id": 1, "seed": 0}')
echo "$RESP"
JOB=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
for i in $(seq 1 40); do
    sleep 15
    OUT=$(curl -s --max-time 5 "http://127.0.0.1:7860/episode/$JOB")
    STATUS=$(echo "$OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
    echo "poll $i: $STATUS"
    if [ "$STATUS" = "done" ] || [ "$STATUS" = "failed" ]; then
        echo "$OUT" | python3 -m json.tool | head -40
        break
    fi
done
