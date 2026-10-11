"""Rootless OSMesa bootstrap for environments without apt root (HF Spaces).

Strategy: `apt-get download` works WITHOUT root (writes .debs to CWD), and
`dpkg-deb -x` extracts without installing. We then preload the shared libs by
absolute path with RTLD_GLOBAL so a later dlopen("libOSMesa.so.8") by name
(mujoco) resolves to our copies — LD_LIBRARY_PATH set after process start would
NOT be honored by glibc, so preloading is the reliable mechanism.

No-op when MUJOCO_GL is already set (local EGL runs) or when a system
libOSMesa loads. Set WEBVLA_FORCE_OSMESA_BOOTSTRAP=1 to force the download
path (used to test this module on machines that already have OSMesa).
"""

import ctypes
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIVE_DIR = os.path.join(REPO_ROOT, "native_libs")

# Best-effort package list; names that don't exist on the image's distro
# release are skipped (apt-get download fails, we log and continue).
APT_PACKAGES = [
    "libosmesa6",
    "libglapi-mesa",
    "libllvm15",   # jammy
    "libllvm14",   # bookworm
    "libllvm17t64",  # noble
    "libllvm19",   # plucky+
    "libxml2",
    "libffi8",
    "libedit2",
    "libzstd1",
    "libtinfo6",
    "zlib1g",
]

# Last-resort direct URLs (Debian bookworm, the HF Spaces base lineage).
DEB_URLS = [
    "http://deb.debian.org/debian/pool/main/m/mesa/libosmesa6_22.3.6-1+deb12u1_amd64.deb",
    "http://deb.debian.org/debian/pool/main/m/mesa/libglapi-mesa_22.3.6-1+deb12u1_amd64.deb",
    "http://deb.debian.org/debian/pool/main/l/llvm-toolchain-14/libllvm14_14.0.6-12_amd64.deb",
]


def _log(msg):
    print(f"[osmesa-bootstrap] {msg}", flush=True)


def _system_osmesa_works():
    try:
        ctypes.CDLL("libOSMesa.so.8")
        return True
    except OSError:
        return False


def _download_debs():
    os.makedirs(NATIVE_DIR, exist_ok=True)
    deb_dir = os.path.join(NATIVE_DIR, "debs")
    os.makedirs(deb_dir, exist_ok=True)
    got = 0
    for pkg in APT_PACKAGES:
        r = subprocess.run(
            ["apt-get", "download", pkg], cwd=deb_dir, capture_output=True, text=True
        )
        if r.returncode == 0:
            got += 1
        else:
            _log(f"apt-get download {pkg}: skipped ({r.stderr.strip()[:80]})")
    if got == 0:
        _log("apt-get download unavailable; falling back to hardcoded bookworm URLs")
        for url in DEB_URLS:
            fn = os.path.join(deb_dir, url.rsplit("/", 1)[-1])
            if not os.path.exists(fn):
                subprocess.run(["curl", "-sSL", "--retry", "3", url, "-o", fn], check=True)
    for fn in os.listdir(deb_dir):
        if fn.endswith(".deb"):
            subprocess.run(
                ["dpkg-deb", "-x", os.path.join(deb_dir, fn), NATIVE_DIR], check=True
            )
    _log(f"extracted {got or len(DEB_URLS)} debs into {NATIVE_DIR}")


def _find_libs():
    libs = []
    for root, _, files in os.walk(NATIVE_DIR):
        for f in files:
            if ".so" in f:
                libs.append(os.path.join(root, f))
    # dependencies first: anything that is not libOSMesa, then libOSMesa last
    libs.sort(key=lambda p: "osmesa" in os.path.basename(p).lower())
    return libs


def _preload():
    loaded = None
    for path in _find_libs():
        try:
            h = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            if "osmesa" in os.path.basename(path).lower():
                loaded = path
        except OSError as e:
            _log(f"preload skip {os.path.basename(path)}: {str(e)[:80]}")
    return loaded


def ensure():
    """Make MUJOCO_GL=osmesa usable; return the backend name in effect."""
    if os.environ.get("MUJOCO_GL") and os.environ.get("MUJOCO_GL") != "osmesa":
        return os.environ["MUJOCO_GL"]
    os.environ["MUJOCO_GL"] = "osmesa"
    if os.environ.get("WEBVLA_FORCE_OSMESA_BOOTSTRAP") != "1" and _system_osmesa_works():
        _log("system libOSMesa found, nothing to do")
        return "osmesa"
    marker = os.path.join(NATIVE_DIR, ".done")
    if not os.path.exists(marker):
        _download_debs()
        open(marker, "w").write("ok")
    loaded = _preload()
    if not loaded:
        raise RuntimeError(
            "OSMesa bootstrap failed: could not load extracted libOSMesa. "
            "Check [osmesa-bootstrap] logs above."
        )
    _log(f"using {loaded}")
    return "osmesa"


if __name__ == "__main__":
    backend = ensure()
    import mujoco  # noqa: F401  (verifies the backend can initialize)
    print(f"OK: MUJOCO_GL={backend}")
    sys.exit(0)
