"""Verify the custom-instruction feature end to end (server must be running).

  bash setup/run.sh setup/test_custom_instruction.py

Checks: custom job -> metadata carries the custom instruction + override flags,
normal job on the same scene is unaffected, gallery flags the custom row, and
the UI badge helper renders UNCHECKED vs SUCCESS/FAILED appropriately.
"""

import os
import sys
import time

import httpx

import paths
from web.ui import _badge_md

BASE = os.environ.get("WEBVLA_API_BASE", "http://127.0.0.1:7860")
CUSTOM = "pick up the ramekin and place it on the plate"


def submit(client, task_id, seed, instruction=None):
    r = client.post(f"{BASE}/episode", json={
        "task_id": task_id, "seed": seed, "instruction": instruction})
    r.raise_for_status()
    return r.json()["job_id"]


def wait(client, job_id, timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = client.get(f"{BASE}/episode/{job_id}").json()
        if d["status"] in ("done", "failed"):
            return d
        time.sleep(5)
    raise TimeoutError(job_id)


def main():
    client = httpx.Client(timeout=30)
    ok = True

    j1 = submit(client, 0, 3, CUSTOM)
    d1 = wait(client, j1)
    m1 = d1.get("result", {}).get("metadata", {})
    good1 = (d1["status"] == "done"
             and m1.get("instruction") == CUSTOM
             and m1.get("instruction_overridden") is True
             and "black bowl" in (m1.get("base_task_language") or ""))
    mp4 = os.path.join(paths.REPO_ROOT, "episodes", m1.get("mp4", ""))
    good1 &= os.path.exists(mp4)
    print(f"[{'OK' if good1 else 'FAIL'}] custom job: instruction + flags in metadata, mp4 exists")
    ok &= good1

    j2 = submit(client, 0, 3)
    d2 = wait(client, j2)
    m2 = d2.get("result", {}).get("metadata", {})
    good2 = (d2["status"] == "done"
             and m2.get("instruction") == m1.get("base_task_language")
             and not m2.get("instruction_overridden"))
    print(f"[{'OK' if good2 else 'FAIL'}] normal job on same scene unaffected")
    ok &= good2

    gallery = client.get(f"{BASE}/episodes").json()["episodes"]
    g1 = [e for e in gallery if e.get("job_id") == j1]
    g2 = [e for e in gallery if e.get("job_id") == j2]
    good3 = g1 and g1[0].get("instruction_overridden") and g2 and not g2[0].get("instruction_overridden")
    print(f"[{'OK' if good3 else 'FAIL'}] gallery flags custom row only")
    ok &= bool(good3)

    b1, b2 = _badge_md(m1), _badge_md(m2)
    good4 = b1.startswith("### CUSTOM INSTRUCTION") and (
        b2.startswith("### SUCCESS") or b2.startswith("### FAILED"))
    print(f"[{'OK' if good4 else 'FAIL'}] badges: custom='{b1.splitlines()[0]}' normal='{b2.splitlines()[0]}'")
    ok &= good4

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
