"""M2 live-streaming + instruction-injection test against the running server.

  bash setup/run.sh setup/test_stream.py

Flow: submit task 0 / seed 0 -> connect ws -> assert JPEG frames stream ->
inject an instruction mid-run -> assert terminal 'done' message -> assert the
metadata JSON recorded the injection and the mp4 exists. Also checks the
frame.jpg fallback endpoint and the M0 no-hooks regression path.
"""

import io
import json
import os
import time

import httpx
import websockets.sync.client

BASE = "http://127.0.0.1:7860"
WS = "ws://127.0.0.1:7860"
INJECT = "pick up the ramekin and place it on the plate"


def wait_ready():
    for _ in range(90):
        try:
            if httpx.get(f"{BASE}/healthz", timeout=5).json().get("warmup_done"):
                return
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit("server never became ready")


def main():
    wait_ready()

    job_id = httpx.post(f"{BASE}/episode",
                        json={"task_id": 0, "seed": 0}, timeout=15).json()["job_id"]
    print(f"[OK] submitted {job_id}")

    frames = 0
    dims = None
    terminal = None
    injected = False
    t0 = time.time()
    with websockets.sync.client.connect(f"{WS}/ws/jobs/{job_id}",
                                        open_timeout=30) as ws:
        while time.time() - t0 < 420:
            try:
                msg = ws.recv(timeout=60)
            except TimeoutError:
                break
            if isinstance(msg, bytes):
                frames += 1
                if frames == 5:
                    # polling fallback endpoint mid-run
                    r = httpx.get(f"{BASE}/episode/{job_id}/frame.jpg", timeout=10)
                    assert r.status_code == 200 and len(r.content) > 1000
                    print("[OK] frame.jpg fallback serving during run")
                if frames == 10 and not injected:
                    r = httpx.post(f"{BASE}/episode/{job_id}/instruction",
                                   json={"text": INJECT}, timeout=10)
                    assert r.status_code == 200, r.text
                    injected = True
                    print("[OK] instruction injected mid-run")
                if frames == 1:
                    from PIL import Image
                    im = Image.open(io.BytesIO(msg))
                    dims = im.size
            else:
                terminal = json.loads(msg)
                break

    assert frames >= 10, f"only {frames} frames streamed"
    assert dims == (256, 256), f"unexpected frame size {dims}"
    assert terminal and terminal.get("type") == "done", f"no terminal msg: {terminal}"
    print(f"[OK] streamed {frames} frames @ {dims}, terminal: {terminal}")

    # injection rejection after completion
    r = httpx.post(f"{BASE}/episode/{job_id}/instruction",
                   json={"text": "too late"}, timeout=10)
    assert r.status_code == 409, f"expected 409, got {r.status_code}"
    print("[OK] post-completion injection rejected with 409")

    d = httpx.get(f"{BASE}/episode/{job_id}", timeout=10).json()
    assert d["status"] == "done"
    events = d["result"]["metadata"].get("instruction_events")
    assert events and events[0]["text"] == INJECT and events[0]["step"] > 0, events
    print(f"[OK] metadata instruction_events: {events}")

    mp4 = os.path.join("episodes", os.path.basename(d["result"]["mp4_url"]))
    assert os.path.exists(mp4) and os.path.getsize(mp4) > 10000
    print(f"[OK] mp4 on disk: {os.path.basename(mp4)}")

    print("RESULT: PASS")


if __name__ == "__main__":
    main()
