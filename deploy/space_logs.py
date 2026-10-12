"""Fetch Space build or run logs.

  HF_TOKEN=... HF_HUB_OFFLINE=0 bash setup/run.sh deploy/space_logs.py [build|run]
"""

import json
import os
import sys
import urllib.request

SPACE = "swn21/webvla-libero-octo"


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "build"
    token = os.environ.get("HF_TOKEN")
    req = urllib.request.Request(
        f"https://huggingface.co/api/spaces/{SPACE}/logs/{kind}",
        headers={"Authorization": f"Bearer {token}"} if token else {},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode("utf-8", "replace")
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        # SSE or plain-text stream: strip "data: " prefixes
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if line and line != "{}":
                print(line)
        return
    if isinstance(d, list):
        for entry in d:
            print(entry.get("message") if isinstance(entry, dict) else entry)
    else:
        print(json.dumps(d)[:20000])


if __name__ == "__main__":
    main()
