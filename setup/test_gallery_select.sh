#!/usr/bin/env bash
# Verify the fixed gallery select handler returns a real mp4 path.
set -e
cd /mnt/c/Users/sng51/Desktop/WebVLA
export WEBVLA_VENV=/root/webvla/venv_lerobot
export WEBVLA_LEROBOT_DEVICE=cuda
cat > /tmp/test_gallery_select.py <<'EOF'
import os
import space.app as a

class FakeEvt:
    index = [0, 0]

rows = a._gallery_rows()
print("gallery rows:", len(rows))
assert rows, "no episodes on disk to select"
p = a._gallery_select(FakeEvt())
print("selected ->", p)
assert p and os.path.exists(p) and p.endswith(".mp4")
# out-of-bounds click must degrade to None, not crash
class BadEvt:
    index = [999, 0]
assert a._gallery_select(BadEvt()) is None
assert a._gallery_select(None) is None
print("PASS")
EOF
exec bash setup/run.sh /tmp/test_gallery_select.py
