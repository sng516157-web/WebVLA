"""Create/update the pi0.5 Gradio-SDK Space and push the app.

Needs HF_TOKEN (write) in the environment.

  HF_TOKEN=... bash setup/run.sh deploy/push_space_pi05.py

Stages a minimal copy of the repo into deploy/_staging_pi05 (gitignored),
pushes it as the Space content, requests ZeroGPU hardware, and sets env vars.

Differences from deploy/push_space.py (the Octo Space): that one is a Docker
SDK Space on cpu-basic with the JAX stack baked into the image. This one is a
Gradio SDK Space — the ONLY way to get ZeroGPU, which is PyTorch-only — so the
app is space/app.py (Gradio UI, no FastAPI), lerobot is pip-installed
--no-deps at boot, and LIBERO + OSMesa are bootstrapped at runtime (no
Dockerfile, no apt).
"""

import os
import shutil

import paths

REPO = paths.REPO_ROOT
STAGING = os.path.join(REPO, "deploy", "_staging_pi05")
SPACE_NAME = os.environ.get("WEBVLA_PI05_SPACE_NAME", "webvla-libero-pi05")

# runtime code only (no setup/, episodes/, results/, deploy/)
COPY_DIRS = ["sim", "policy", "server", os.path.join("assets", "scenes")]
COPY_FILES = ["paths.py"]
RENAMES = {
    os.path.join("space", "app.py"): "app.py",
    os.path.join("space", "bootstrap_libero.py"): "bootstrap_libero.py",
    os.path.join("space", "requirements.txt"): "requirements.txt",
    os.path.join("space", "README_space.md"): "README.md",
    os.path.join("space", "xray_overlay.py"): "xray_overlay.py",
    os.path.join("space", "xray_panels.py"): "xray_panels.py",
}

# PaliGemma sentencepiece vocab — license-gated on HF, so we ship the file
# (openpi's assets cache copy; token ids verified identical to the gated
# tokenizer). policy/lerobot_pi05_policy.py picks it up via
# WEBVLA_PALIGEMMA_TOKENIZER, which app.py sets to this path by default.
TOKENIZER_SRC = os.environ.get(
    "WEBVLA_PALIGEMMA_TOKENIZER",
    "/root/.cache/openpi/big_vision/paligemma_tokenizer.model",
)


def stage():
    if os.path.exists(STAGING):
        shutil.rmtree(STAGING)
    os.makedirs(STAGING)
    for d in COPY_DIRS:
        shutil.copytree(
            os.path.join(REPO, d), os.path.join(STAGING, d),
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    for f in COPY_FILES:
        shutil.copy2(os.path.join(REPO, f), os.path.join(STAGING, f))
    for src, dst in RENAMES.items():
        shutil.copy2(os.path.join(REPO, src), os.path.join(STAGING, dst))
    # server/ is copied wholesale (server/app.py needs fastapi etc., which the
    # Space never installs) — trim it to what space/app.py actually imports.
    for fn in os.listdir(os.path.join(STAGING, "server")):
        if fn not in ("__init__.py", "bootstrap_osmesa.py"):
            p = os.path.join(STAGING, "server", fn)
            shutil.rmtree(p) if os.path.isdir(p) else os.unlink(p)
    if not os.path.exists(TOKENIZER_SRC):
        raise SystemExit(
            f"paligemma tokenizer not found at {TOKENIZER_SRC} — the Space "
            "cannot load pi0.5 without it (google/paligemma-3b-pt-224 is "
            "license-gated). Set WEBVLA_PALIGEMMA_TOKENIZER."
        )
    shutil.copy2(TOKENIZER_SRC, os.path.join(STAGING, "paligemma_tokenizer.model"))
    print(f"staged {STAGING}: " + ", ".join(sorted(os.listdir(STAGING))))


def main():
    from huggingface_hub import HfApi, SpaceHardware

    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    space_id = f"{user}/{SPACE_NAME}"

    api.create_repo(space_id, repo_type="space", space_sdk="gradio",
                    exist_ok=True)
    stage()
    api.upload_folder(
        folder_path=STAGING, repo_id=space_id, repo_type="space",
        ignore_patterns=["**/__pycache__/**"],
    )
    print(f"pushed -> https://huggingface.co/spaces/{space_id}")

    # ZeroGPU is the whole point of this Space (CPU inference measured at
    # ~100 s/chunk — unusable). Requires PRO on the account; if the API call
    # fails, flip it manually in Settings -> Hardware -> ZeroGPU.
    try:
        api.request_space_hardware(space_id, SpaceHardware.ZERO_A10G)
        print("requested ZeroGPU (zero-a10g) hardware")
    except Exception as e:
        print(f"hardware request failed ({e}) — set ZeroGPU manually in the "
              f"Space settings before announcing the URL")


if __name__ == "__main__":
    main()
