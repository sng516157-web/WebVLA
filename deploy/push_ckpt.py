"""One-shot: upload the fine-tuned 50k checkpoint to the HF model hub.

Needs HF_TOKEN (write) in the environment. Uploads only what
OctoModel.load_pretrained needs: config.json, dataset_statistics.json,
example_batch.msgpack, finetune_config.json, and the 50000/ step dir (~773MB).

  HF_TOKEN=... bash setup/run.sh deploy/push_ckpt.py
"""

import os

import paths

CKPT_DIR = os.environ.get(
    "WEBVLA_FT_CKPT",
    os.path.join(
        paths.CKPTS_DIR, "octo_finetune", "libero_spatial",
        "libero_spatial_bs16_lr5e5_20261010_114251",
    ),
)
REPO_NAME = "octo-libero-spatial-ft"
ALLOW = [
    "config.json",
    "dataset_statistics.json",
    "example_batch.msgpack",
    "finetune_config.json",
    "50000/**",
]


def main():
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    repo_id = f"{user}/{REPO_NAME}"
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=CKPT_DIR,
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=ALLOW,
    )
    print(f"uploaded {CKPT_DIR} -> hf://{repo_id}")


if __name__ == "__main__":
    main()
