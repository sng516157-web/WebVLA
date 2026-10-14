#!/usr/bin/env bash
# True browser verification of gallery select on the LIVE Space:
# 1) record one episode via the API (fresh containers start with an empty
#    gallery), 2) drive headless Chromium: load the page, click the first
#    gallery row, assert the "selected episode" <video> gets an mp4 src.
set -e
source ~/webvla-env.sh
: "${HF_TOKEN:?set HF_TOKEN in the environment}"
cat > /tmp/test_space_gallery_browser.py <<'PYEOF'
import os
import time

BASE = "https://swn21-webvla-libero-pi05.hf.space"
TOKEN = os.environ["HF_TOKEN"]

from gradio_client import Client
c = Client(BASE, token=TOKEN)
print("recording one episode ...", flush=True)
video, frame, status, badge = c.predict(0, 0, "", api_name="/run_episode")
print("episode:", badge.splitlines()[0], flush=True)

from playwright.sync_api import sync_playwright

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page()
    page.goto(BASE, wait_until="domcontentloaded")
    # gallery rows populate via demo.load; Gradio 6 renders the Dataframe
    # as a virtualized role=grid (data rows are divs, not <tr>), so wait
    # for a data row (aria-rowindex >= 2) instead of table cells
    page.wait_for_selector("div[role=grid] [aria-rowindex='2']",
                           state="attached", timeout=120000)
    time.sleep(3)  # let hydration finish
    row = page.locator("div[role=grid] [aria-rowindex='2']").first
    print("gallery row 0:", row.inner_text()[:100].replace("\n", " | "))
    print("bounding box:", row.bounding_box())
    row.scroll_into_view_if_needed()
    row.click()
    # the selected-episode player is a <video> whose src gains an mp4
    page.wait_for_function(
        "Array.from(document.querySelectorAll('video')).some("
        "  v => v.src && v.src.includes('.mp4'))",
        timeout=30000)
    src = page.evaluate(
        "Array.from(document.querySelectorAll('video')).map(v => v.src)")
    print("video elements:", src)
    assert any(".mp4" in s for s in src)
    browser.close()
print("PASS: clicking a gallery row loads the episode video in a real browser")
PYEOF
/root/webvla/venv_lerobot/bin/python /tmp/test_space_gallery_browser.py
