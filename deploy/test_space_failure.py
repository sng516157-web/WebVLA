"""Induce a failure on the live Space (invalid task_id) and confirm the
server stays healthy afterwards.

  HF_HUB_OFFLINE=0 bash setup/run.sh deploy/test_space_failure.py
"""

import json
import time
import urllib.request

BASE = "https://swn21-webvla-libero-octo.hf.space"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.load(r)


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    job_id = post("/episode", {"task_id": 999, "seed": 0})["job_id"]
    print("submitted bad job:", job_id, flush=True)
    for _ in range(30):
        d = get(f"/episode/{job_id}")
        if d["status"] in ("done", "failed"):
            break
        time.sleep(10)
    print("status:", d["status"])
    assert d["status"] == "failed", f"expected failed, got {d['status']}"
    err = d.get("error") or ""
    print("error tail:", err[-300:])
    h = get("/healthz")
    print("healthz after failure:", h)
    assert h["ok"] and h["worker_alive"], "server/worker did not survive"
    print("PASS: failed episode handled, server alive")


if __name__ == "__main__":
    main()
