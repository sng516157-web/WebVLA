#!/bin/bash
# Poll the Space runtime stage until it leaves BUILDING (or errors). ~55 min cap.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
for i in $(seq 1 55); do
  s=$(HF_HUB_OFFLINE=0 bash setup/run.sh deploy/space_status.py 2>/dev/null)
  echo "[$i] $s"
  case "$s" in
    RUNNING\ *|RUNNING|*ERROR*|NO_APP*|CONFIG_ERROR*|PAUSED*) exit 0;;
  esac
  sleep 60
done
