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
    _patch_weights_only(TARGET)
    sys.path.insert(0, TARGET)
    print(f"[libero-bootstrap] LIBERO @ {TARGET}", flush=True)
    return TARGET


def _patch_weights_only(target):
    """torch>=2.6 defaults torch.load(weights_only=True); LIBERO's init-state
    files are pickled numpy and need False. The tarball is pristine, so apply
    the same patch our local LIBERO checkout carries. Trusted local data."""
    bench = os.path.join(target, "libero", "libero", "benchmark", "__init__.py")
    with open(bench) as f:
        src = f.read()
    old = "torch.load(init_states_path)"
    new = "torch.load(init_states_path, weights_only=False)"
    if old in src and new not in src:
        with open(bench, "w") as f:
            f.write(src.replace(old, new))
        print("[libero-bootstrap] patched weights_only=False in benchmark/__init__.py",
              flush=True)
