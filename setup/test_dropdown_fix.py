"""Verify the dropdown-retry fix: poll() with an empty choices state must
fetch tasks from the API and return a dropdown update with choices + value.
Requires the local server running (setup/start_server.sh or the task).
"""

import time

import httpx

from web.ui import poll

for _ in range(60):
    try:
        if httpx.get("http://127.0.0.1:7860/tasks", timeout=5).json().get("ready"):
            break
    except Exception:
        pass
    time.sleep(5)
else:
    raise SystemExit("server /tasks never became ready")

out = poll(None, 1, [])
# returns (job, status_md, video, stats, error, gallery, dd, choices)
choices_state = out[7]
dd_update = out[6]
assert choices_state, "choices state still empty"
assert len(choices_state) == 10, f"expected 10 tasks, got {len(choices_state)}"
assert choices_state[0][1] == 0
print("PASS: dropdown choices load on first poll:", choices_state[0][0][:60], "...")

# second poll with populated state must not override the user's selection
out2 = poll(None, 2, choices_state)
assert out2[7] == choices_state
dd2 = out2[6]
val = getattr(dd2, "value", "sentinel") if not isinstance(dd2, dict) else dd2.get("value", "sentinel")
print("PASS: second poll keeps state; dd update value:", val)
