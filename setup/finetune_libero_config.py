"""Octo finetune config for LIBERO libero_spatial (RLDS converted dataset).

Deviations from octo's default finetune_config.py (hardware-driven, documented):
  - batch_size 256 -> 16 (single RTX 4080 Laptop 12GB)
  - num_steps 50000 -> 50000, batch-16 (see below)
  - lr peak 3e-4 -> 5e-5: 3e-4 is tuned for batch 256; at batch 16 the
    10k-step run diverged (action MSE 0.67 vs GT var ~0.2, worse than
    predicting the mean). 5e-5 sits between linear (1.9e-5) and sqrt (7.5e-5)
    scaling of the default.
  - warmup 2000 -> 500 (shorter run, small batch)
  - viz eval_batch_size 128 -> 32 (VRAM headroom)
  - shuffle_buffer 10000 -> 500, frame threads 16 -> 6 (host RAM: the 15GB
    WSL VM OOM-killed the run with the default buffer of decoded frames)
Everything else (full finetune, window_size 1, action_horizon 4, augmentation,
resize 256/128) is the octo default.

Usage (WSL):
  bash setup/run_finetune.sh   (resolves paths via paths.py / WEBVLA_* env vars)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import CKPTS_DIR, OCTO_DIR, RLDS_DIR

sys.path.insert(0, OCTO_DIR)

from scripts.configs.finetune_config import get_config as _base_get_config


def get_config(config_string="full,language_conditioned"):
    config = _base_get_config(config_string)

    config.pretrained_path = "hf://rail-berkeley/octo-base-1.5"

    config.dataset_kwargs = dict(
        name="libero_rlds",  # tfds builder class LiberoRlds -> dataset name
        data_dir=RLDS_DIR,
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
    config.shuffle_buffer_size = 500
    config["frame_transform_threads"] = 6
    config.num_steps = 50000
    config.optimizer.learning_rate.peak_value = 5e-5
    config.optimizer.learning_rate.warmup_steps = 500
    config.optimizer.learning_rate.decay_steps = 50000
    config.save_interval = 5000
    config.eval_interval = 5000
    config.save_dir = CKPTS_DIR

    config.viz_kwargs.eval_batch_size = 32
    config.viz_kwargs.trajs_for_metrics = 50
    config.viz_kwargs.trajs_for_viz = 4

    config.wandb.project = "octo_finetune"
    config.wandb.group = "libero_spatial"
    config.wandb.entity = "webvla-local"
    return config
