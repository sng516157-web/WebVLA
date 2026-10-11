"""Create/update the HF Space and push the app.

Needs HF_TOKEN (write) in the environment.

  HF_TOKEN=... bash setup/run.sh deploy/push_space.py

Stages a minimal copy of the repo into deploy/_staging (gitignored), pushes it
as the Space content, then requests ZeroGPU hardware and sets the env vars.
The Space is a Gradio SDK space: HF installs requirements.txt and runs app.py.
"""

import os
import shutil

import paths

REPO = paths.REPO_ROOT
STAGING = os.path.join(REPO, "deploy", "_staging")
SPACE_NAME = os.environ.get("WEBVLA_SPACE_NAME", "webvla-libero-octo")

# what the Space needs: runtime code only (no setup/, episodes/, results)
COPY_DIRS = ["server", "web", "runner", "sim", "policy", "assets"]
COPY_FILES = ["app.py", "paths.py"]
RENAMES = {
    "requirements-space.txt": "requirements.txt",
    os.path.join("deploy", "space_readme.md"): "README.md",
}


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
    print(f"staged {STAGING}: "
          + ", ".join(sorted(os.listdir(STAGING))))


def main():
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    space_id = f"{user}/{SPACE_NAME}"

    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    stage()
    api.upload_folder(
        folder_path=STAGING, repo_id=space_id, repo_type="space",
        ignore_patterns=["**/__pycache__/**"],
    )
    print(f"pushed -> https://huggingface.co/spaces/{space_id}")

    # Hardware: cpu-basic (default) on purpose. ZeroGPU ("zero-a10g") would only
    # help if we shipped CUDA jaxlib — but the pinned jaxlib 0.4.20+cudnn89
    # predates ZeroGPU's Blackwell (sm_120) hardware and cannot run on it, and
    # ZeroGPU officially supports PyTorch only. CPU inference measured at
    # 1.23s/replan on 2 cores (setup/bench_cpu_latency.py) -> ~2-4 min episodes.
    # To experiment later: request SpaceHardware.ZERO_A10G here and decorate the
    # rollout with @spaces.GPU, expecting to document the failure mode.

    for k, v in {"WEBVLA_BACKEND": "cpu"}.items():
        try:
            api.add_space_variable(space_id, k, v)
        except Exception as e:
            print(f"variable {k} failed ({e})")


if __name__ == "__main__":
    main()
