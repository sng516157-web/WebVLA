#!/usr/bin/env bash
# End-to-end browser test of live-mode X-Ray (localhost:7870).
# Prereq: server already running:
#   WEBVLA_VENV=/root/webvla/venv_lerobot WEBVLA_POLICY=lerobot_pi05 \
#     WEBVLA_LEROBOT_DEVICE=cuda bash setup/run.sh -m uvicorn live.app:app --port 7870
# Drives headless Chromium: enable all overlays, start task 0 seed 0, let it
# run ~45s, assert every xray envelope type arrived, screenshot the stage,
# click the video (push), stop without success -> assert the failure bundle
# event + files on disk.
set -e
source ~/webvla-env.sh 2>/dev/null || true
cat > /tmp/test_live_xray.py <<'PYEOF'
import json
import os
import time
import urllib.request

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:7870"
SHOTS = "/mnt/c/Users/sng51/Desktop/WebVLA/assets/xray"

for _ in range(180):  # model load can take minutes
    try:
        if json.load(urllib.request.urlopen(BASE + "/tasks", timeout=2))["ready"]:
            break
    except Exception:
        pass
    time.sleep(2)
else:
    raise SystemExit("server never became ready")
print("server ready", flush=True)

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(BASE, wait_until="domcontentloaded")
    page.wait_for_selector("#task option", state="attached", timeout=30000)
    # record every envelope type seen
    page.evaluate("""() => {
      window.__xrayLog = {};
      const orig = Director.on.bind(Director);
      ['xray.chunk','xray.exec','xray.denoise','xray.heatmap','xray.bundle',
       'frame.meta','click.result','session.state','error'].forEach(t =>
        orig(t, env => { window.__xrayLog[t] = (window.__xrayLog[t]||0) + 1; }));
    }""")
    # enable every overlay
    for cb in page.locator("[data-xray]").all():
        cb.check()
    # start episode: task 0, seed 0
    page.select_option("#task", "0")
    page.click("#start")
    print("episode started, running 45s ...", flush=True)
    time.sleep(45)
    page.locator("#stage").screenshot(path=f"{SHOTS}/live_all_overlays.png")
    # click near the middle of the scene (continuous command envelope)
    page.mouse.click(320, 300)
    time.sleep(3)
    # stop; task 0 seed 0 often SUCCEEDS within 45s on GPU, so the bundle
    # (failure-only by design) is tested on a second, deliberately short run
    page.click("#stop")
    time.sleep(4)
    page.click("#start")   # same task/seed; 10s cannot succeed
    time.sleep(10)
    page.click("#stop")
    time.sleep(4)
    log = page.evaluate("window.__xrayLog")
    print("envelope counts:", json.dumps(log, indent=1))
    for t in ("frame.meta", "xray.chunk", "xray.exec", "xray.denoise",
              "xray.heatmap", "click.result", "xray.bundle"):
        assert log.get(t, 0) > 0, f"no {t} envelopes seen"
    assert "error" not in log, f"server errors: {log['error']}"
    browser.close()

import glob
bundles = glob.glob("/mnt/c/Users/sng51/Desktop/WebVLA/live/xray_bundles/*")
assert bundles, "no failure bundle written"
latest = max(bundles, key=os.path.getmtime)
idx = json.load(open(os.path.join(latest, "index.json")))
print("bundle:", latest, "| slots:", idx["n_replans"], "| reason:", idx["reason"])
assert idx["n_replans"] > 0
print("PASS: live X-Ray e2e — all overlays streaming, click path, bundle flush")
PYEOF
/root/webvla/venv_lerobot/bin/python /tmp/test_live_xray.py
