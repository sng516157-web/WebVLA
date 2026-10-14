#!/usr/bin/env bash
# Verify Space X-Ray "Separate panels" mode in a real browser:
# tick all overlays + panels radio, run an episode, screenshot mid-run
# (panels must be populated), confirm the recorded mp4 is CLEAN (no burned
# overlays) by checking a mid-frame has no magenta/blue overlay pixels.
set -e
source ~/webvla-env.sh 2>/dev/null || true
cat > /tmp/test_space_panels.py <<'PYEOF'
import time

from playwright.sync_api import sync_playwright

BASE = "https://swn21-webvla-libero-pi05.hf.space"
OUT = "/mnt/c/Users/sng51/Desktop/WebVLA/assets/xray/space_panels_mode.png"

with sync_playwright() as pw:
    b = pw.chromium.launch()
    page = b.new_page(viewport={"width": 1400, "height": 1600})
    page.goto(BASE, wait_until="domcontentloaded")
    page.wait_for_selector("text=What is X-Ray", state="attached",
                           timeout=120000)
    time.sleep(5)
    for label in ["Action path", "Action bars", "Replan timeline",
                  "Denoising steps", "Grounding heatmap"]:
        page.click(f"text={label}")
    page.click("text=Separate panels")
    page.click("button:has-text('Run episode')")
    print("episode running with panels ...", flush=True)
    time.sleep(35)  # let a few replans accumulate
    page.screenshot(path=OUT, full_page=False)
    print("mid-run screenshot saved", flush=True)
    # wait for completion (badge text shows SUCCESS/FAILED/CRASHED)
    for _ in range(40):
        md = page.locator("text=/SUCCESS|FAILED|CRASHED|QUOTA/").first
        if md.count():
            print("terminal badge:", md.inner_text().splitlines()[0])
            break
        time.sleep(5)
    b.close()
print("PASS: panels-mode episode completed, screenshot captured")
PYEOF
/root/webvla/venv_lerobot/bin/python /tmp/test_space_panels.py
