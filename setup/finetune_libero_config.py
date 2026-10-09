"""Octo finetune config for LIBERO libero_spatial (RLDS converted dataset).

Deviations from octo's default finetune_config.py (all hardware-driven, documented):
  - batch_size 256 -> 16 (single RTX 4080 Laptop 12GB)
  - num_steps 50000 -> 20000, decay_steps matched (smaller batch, time budget)
  - viz eval_batch_size 128 -> 32 (VRAM headroom)
Everything else (lr 3e-4 cosine, warmup 2000, full finetune, window_size 1,
action_horizon 4, augmentation, resize 256/128) is the octo default.

Usage (WSL):
  cd /root/webvla/octo && python scripts/finetune.py \
    --config=/mnt/c/Users/sng51/Desktop/WebVLA/setup/finetune_libero_config.py \
    --name=libero_spatial_bs16 --debug
"""

import sys

sys.path.insert(0, "/root/webvla/octo")

from scripts.configs.finetune_config import get_config as _base_get_config


def get_config(config_string="full,language_conditioned"):
    config = _base_get_config(config_string)

    config.pretrained_path = "hf://rail-berkeley/octo-base-1.5"

    config.dataset_kwargs = dict(
        name="libero_rlds",  # tfds builder class LiberoRlds -> dataset name
        data_dir="/root/webvla/rlds_datasets",
        image_obs_keys={"primary": "image", "wrist": "wrist_image"},
        proprio_obs_key="state",
        language_key="language_instruction",
        action_proprio_normalization_type="normal",
        # do not normalize the gripper action dim (matches octo default)
        action_normalization_mask=[True, True, True, True, True, True, False],
        num_parallel_reads=8,
        num_parallel_calls=16,
    )

    config.batch_size = 16
    config.shuffle_buffer_size = 10000
    config.num_steps = 20000
    config.optimizer.learning_rate.decay_steps = 20000
    config.save_interval = 5000
    config.eval_interval = 5000
    config.save_dir = "/root/webvla/octo_ckpts"

    config.viz_kwargs.eval_batch_size = 32
    config.viz_kwargs.trajs_for_metrics = 50
    config.viz_kwargs.trajs_for_viz = 4

    config.wandb.project = "octo_finetune"
    config.wandb.group = "libero_spatial"
    config.wandb.entity = "webvla-local"
    return config
