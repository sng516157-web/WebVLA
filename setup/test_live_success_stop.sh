#!/usr/bin/env bash
# Verify the success-stop fix: start task 0 seed 0 (known to succeed),
# poll /session until the episode ends BY ITSELF, assert success + idle.
set -e
bash /mnt/c/Users/sng51/Desktop/WebVLA/setup/wait_live_ready.sh
sleep 15  # let the policy finish loading (tasks ready fires on wrapper only)
curl -s -X POST http://127.0.0.1:7870/session/start \
  -H "Content-Type: application/json" -d '{"task_id": 0, "seed": 0}'
echo
for i in $(seq 1 60); do
  s=$(curl -s --max-time 3 http://127.0.0.1:7870/session)
  state=$(echo "$s" | grep -o '"state":"[a-z]*"' | head -1)
  succ=$(echo "$s" | grep -o '"success":[a-z]*' | head -1)
  steps=$(echo "$s" | grep -o '"steps":[0-9]*' | head -1)
  echo "poll $i: $state $succ $steps"
  case "$state" in
    *idle*)
      case "$succ" in
        *true*) echo "PASS: episode self-terminated on success ($steps)"; exit 0;;
        *) echo "ended but not success ($succ)"; exit 1;;
      esac;;
  esac
  sleep 5
done
echo "FAIL: still running after 5 min"; exit 1
