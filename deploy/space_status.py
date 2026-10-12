"""Print the Space's runtime stage (and hardware) as one line.

  HF_HUB_OFFLINE=0 bash setup/run.sh deploy/space_status.py
"""

import json
import urllib.request

SPACE = "swn21/webvla-libero-octo"


def main():
    with urllib.request.urlopen(
        f"https://huggingface.co/api/spaces/{SPACE}", timeout=30
    ) as r:
        d = json.load(r)
    rt = d.get("runtime") or {}
    print(rt.get("stage"), "| hw:", (rt.get("hardware") or {}).get("current"))


if __name__ == "__main__":
    main()
