"""Runtime LIBERO install for the Gradio SDK Space (no Dockerfile, no apt).

pip-installing LIBERO drops its data files (.bddl / .init / assets) because
the repo ships no MANIFEST.in — so we fetch the pinned source tarball, extract
it under the user's cache dir, put it on sys.path, and pre-write the package's
~/.libero/config.yaml (otherwise first import prompts interactively).

Idempotent; a no-op after the first boot of a container.
"""

import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
TARBALL_URL = ("https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/"
               f"tar.gz/{LIBERO_COMMIT}")
TARGET = os.path.expanduser("~/.cache/webvla/LIBERO")


def ensure():
    if "libero" in sys.modules:
        return TARGET
    import importlib.util
    if importlib.util.find_spec("libero") is not None:
        return None  # local dev: LIBERO already installed (venv .pth)
    if not os.path.isdir(os.path.join(TARGET, "libero")):
        os.makedirs(os.path.dirname(TARGET), exist_ok=True)
        print(f"[libero-bootstrap] downloading {TARBALL_URL}", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
            urllib.request.urlretrieve(TARBALL_URL, tf.name)
            with tarfile.open(tf.name) as tar:
                tar.extractall(os.path.dirname(TARGET),
                               filter="data")
        os.rename(os.path.join(os.path.dirname(TARGET),
                               f"LIBERO-{LIBERO_COMMIT}"), TARGET)
        os.unlink(tf.name)
    config_dir = os.path.expanduser("~/.libero")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "config.yaml"), "w") as f:
        f.write(
            f"benchmark_root: {TARGET}/libero/libero\n"
            f"bddl_files: {TARGET}/libero/libero/bddl_files\n"
            f"init_states: {TARGET}/libero/libero/init_files\n"
            f"datasets: {TARGET}/libero/datasets\n"
            f"assets: {TARGET}/libero/libero/assets\n"
        )
    sys.path.insert(0, TARGET)
    print(f"[libero-bootstrap] LIBERO @ {TARGET}", flush=True)
    return TARGET
