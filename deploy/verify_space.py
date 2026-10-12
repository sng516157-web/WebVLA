"""End-to-end verification of the live Space.

  HF_HUB_OFFLINE=0 bash setup/run.sh deploy/verify_space.py [--no-submit]

Checks /healthz and /tasks; unless --no-submit, submits one episode
(task 0, seed 0), polls to completion, and verifies the mp4 serves.
"""

import json
import sys
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
    print("healthz:", get("/healthz"))
    tasks = get("/tasks")
    print("tasks ready:", tasks.get("ready"), "| n_tasks:", len(tasks.get("tasks", [])))
    if "--no-submit" in sys.argv:
        return

    job_id = post("/episode", {"task_id": 0, "seed": 0})["job_id"]
    print("submitted job:", job_id, flush=True)
    t0 = time.time()
    while True:
        d = get(f"/episode/{job_id}")
        st = d["status"]
        el = int(time.time() - t0)
        print(f"[{el:4d}s] {st} pos={d.get('queue_position')}", flush=True)
        if st == "done":
            m = d["result"]["metadata"]
            print("RESULT:", json.dumps({k: m.get(k) for k in (
                "success", "steps", "wall_time_s", "latency_mean_ms",
                "instruction")}, indent=1))
            url = BASE + d["result"]["mp4_url"]
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as r:
                print("mp4:", r.status, r.headers.get("Content-Type"),
                      r.headers.get("Content-Length"), "bytes")
            break
        if st == "failed":
            print("FAILED:", (d.get("error") or "")[-2000:])
            sys.exit(1)
        if el > 1800:
            print("TIMEOUT waiting for episode")
            sys.exit(1)
        time.sleep(20)


if __name__ == "__main__":
    main()
