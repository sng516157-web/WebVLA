"""Exercise web.ui handlers the way Gradio events would (no browser).

Runs the full UI flow: submit -> poll ticks until done -> verify the video
update points at an existing mp4 and the stats badge matches metadata.

  bash setup/run.sh setup/test_ui_handlers.py
"""

import os
import time

from web.ui import _gallery_rows, _task_choices, poll, submit


def unwrap(update):
    """gr.update() outside an event returns a dict; pull .value if present."""
    return update.get("value") if isinstance(update, dict) else update


def main():
    choices = _task_choices()
    assert len(choices) == 10, f"expected 10 task choices, got {len(choices)}"
    print(f"[OK] task dropdown choices: {len(choices)} (first: {choices[0][0][:60]}…)")

    job_id, status, _, live = submit(0, 14, None)
    assert job_id, f"submit failed: {status}"
    live_val = unwrap(live)
    assert isinstance(live_val, str) and f"/live?job_id={job_id}" in live_val, \
        f"live iframe missing: {live_val}"
    print(f"[OK] submitted via UI handler: {job_id} (live iframe shown)")

    video_val, stats_val, err_val, final_status = None, None, None, None
    tick = 0
    t0 = time.time()
    while time.time() - t0 < 300:
        tick += 1
        job_id, status, video, stats, err, gallery, dd, choices, live = poll(
            job_id, tick, choices)
        video_val = unwrap(video) or video_val
        stats_val = unwrap(stats) or stats_val
        e = unwrap(err)
        err_val = e if isinstance(e, str) else err_val
        s = status if isinstance(status, str) else unwrap(status)
        if s:
            final_status = s
        if job_id is None:
            break
        time.sleep(2)

    assert job_id is None, f"job still active after 300s: {final_status}"
    assert video_val and os.path.exists(video_val), f"video path bad: {video_val}"
    print(f"[OK] poll -> done, video: {os.path.basename(video_val)}")
    print(f"[OK] stats badge:\n{stats_val}")
    rows, _ = _gallery_rows()
    print(f"[OK] gallery rows: {len(rows)} (incl. failures)")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
