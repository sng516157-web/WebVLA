"""DoD curl-test for the M1 API (works against local server or the Space).

  bash setup/run.sh setup/test_api.py --base http://127.0.0.1:7860          # failure-recovery only (~2 min)
  bash setup/run.sh setup/test_api.py --base http://127.0.0.1:7860 --full   # + 3 sequential jobs (~5 min)

Asserts:
  --full: 3 jobs run strictly sequentially (chained started/finished times)
  an invalid task_id job ends status=failed WITH a traceback
  the next valid job still completes (worker not poisoned by the failure)
  /episodes gallery contains the failure record
"""

import argparse
import json
import sys
import time

import httpx


def submit(client, base, task_id, seed):
    r = client.post(f"{base}/episode", json={"task_id": task_id, "seed": seed})
    r.raise_for_status()
    return r.json()["job_id"]


def wait(client, base, job_id, timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"{base}/episode/{job_id}")
        r.raise_for_status()
        d = r.json()
        if d["status"] in ("done", "failed"):
            return d
        time.sleep(5)
    raise TimeoutError(f"{job_id} still {d['status']} after {timeout}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:7860")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    base = args.base.rstrip("/")
    client = httpx.Client(timeout=30)
    ok = True

    if args.full:
        t0 = time.time()
        jobs = [submit(client, base, t, s) for t, s in [(0, 10), (1, 11), (2, 12)]]
        finals = [wait(client, base, j) for j in jobs]
        chained = all(
            abs((finals[i + 1]["started_at"] or 0) - finals[i]["finished_at"]) < 15
            for i in range(2)
        )
        print(f"[{'OK' if chained else 'FAIL'}] 3 jobs sequential "
              f"({time.time()-t0:.0f}s, intervals chained: {chained})")
        ok &= chained

    bad = submit(client, base, 99, 0)
    good = submit(client, base, 0, 13)
    bad_final = wait(client, base, bad)
    has_tb = bad_final["status"] == "failed" and "Traceback" in (bad_final.get("error") or "")
    print(f"[{'OK' if has_tb else 'FAIL'}] invalid task -> failed with traceback")
    ok &= has_tb

    good_final = wait(client, base, good)
    recovered = good_final["status"] == "done" and "result" in good_final
    print(f"[{'OK' if recovered else 'FAIL'}] next valid job completed after failure "
          f"(success={good_final.get('result', {}).get('metadata', {}).get('success')})")
    ok &= recovered

    gallery = client.get(f"{base}/episodes").json()["episodes"]
    in_gallery = any(e.get("job_id") == bad and e.get("error") for e in gallery)
    print(f"[{'OK' if in_gallery else 'FAIL'}] failure visible in /episodes gallery")
    ok &= in_gallery

    alive = client.get(f"{base}/healthz").json()
    print(f"[{'OK' if alive['worker_alive'] else 'FAIL'}] worker alive: {alive}")
    ok &= alive["worker_alive"]

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
