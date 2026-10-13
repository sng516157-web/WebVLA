"""Local end-to-end smoke test for the ZeroGPU Space app (space/app.py).

Assumes the app is already serving on 127.0.0.1:7860 (started with a short
WEBVLA_SPACE_MAX_STEPS, e.g. 10). Submits one episode through the Gradio API
— the same code path a browser visitor hits — and verifies the final yield
carries a playable mp4 plus status text, and that metadata landed in
episodes/.

  bash setup/run.sh deploy/test_space_app_local.py
"""

import glob
import json
import os
import sys

import paths


def main():
    from gradio_client import Client

    before = set(glob.glob(os.path.join(paths.REPO_ROOT, "episodes", "*.json")))
    c = Client("http://127.0.0.1:7860")
    print("submitting episode (task 0, seed 0) via Gradio API ...", flush=True)
    job = c.submit(0, 0, "", api_name="/run_episode")
    video, frame, status, badge = job.result()
    print(f"status: {status}")
    print(f"badge:\n{badge}")
    assert video, "no video returned"
    print(f"video: {video}")

    after = set(glob.glob(os.path.join(paths.REPO_ROOT, "episodes", "*.json")))
    new = sorted(after - before)
    assert new, "no new metadata JSON written to episodes/"
    with open(new[-1]) as f:
        meta = json.load(f)
    print(f"metadata: {new[-1]}")
    print(f"  steps={meta['steps']} success={meta['success']} "
          f"replans={meta.get('n_replans')} lat_mean={meta.get('latency_mean_ms')}")
    assert meta["steps"] > 0 and meta.get("n_replans", 0) > 0
    assert os.path.exists(os.path.join(paths.REPO_ROOT, "episodes", meta["mp4"]))
    print("PASS")


if __name__ == "__main__":
    sys.exit(main())
