#!/usr/bin/env bash
# Debug: what does the live Space DOM actually contain for the gallery?
set -e
source ~/webvla-env.sh
cat > /tmp/debug_space_dom.py <<'PYEOF'
import time
from playwright.sync_api import sync_playwright

BASE = "https://swn21-webvla-libero-pi05.hf.space"
with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page()
    page.goto(BASE, wait_until="domcontentloaded")
    time.sleep(20)
    print("title:", page.title())
    print("tables:", page.locator("table").count())
    print("rows:", page.locator("tr").count())
    print("videos:", page.locator("video").count())
    print("buttons:", page.locator("button").count())
    # any gradio error overlay?
    print("body snippet:", page.locator("body").inner_text()[:400])
    page.screenshot(path="/tmp/space_debug.png", full_page=True)
    browser.close()
PYEOF
/root/webvla/venv_lerobot/bin/python /tmp/debug_space_dom.py
