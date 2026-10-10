"""Diagnostic: does the finetuned checkpoint predict demo actions?

Reuses the production OctoPolicy path (window inference, stack_and_pad,
proprio normalization, unnormalization) on val trajectories.
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import tensorflow_datasets as tfds
import cv2
from paths import CKPTS_DIR, RLDS_DIR
from policy.octo_policy import OctoPolicy

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    os.environ.get("WEBVLA_FT_CKPT", CKPTS_DIR)
policy = OctoPolicy(checkpoint=CKPT)
print("window_size:", policy.window_size, "unnorm_key:", policy.unnorm_key,
      "proprio_norm:", policy._proprio_stats is not None)

ds = tfds.load("libero_rlds", data_dir=RLDS_DIR, split="val")

all_sq, n = 0.0, 0
trace_printed = False
for ep in ds.take(4):
    steps = list(ep["steps"])
    lang = steps[0]["language_instruction"].numpy().decode()
    policy.reset(lang)
    preds, gts = [], []
    for s in steps:
        obs = {
            # RLDS stores 128x128; training frame_transform upsizes primary to 256
            "image_primary": cv2.resize(s["observation"]["image"].numpy(), (256, 256)),
            "image_wrist": s["observation"]["wrist_image"].numpy(),
            "proprio": s["observation"]["state"].numpy().astype(np.float32),
        }
        act = policy.act(obs)[0]  # first action of the chunk
        gt = s["action"].numpy()
        preds.append(act)
        gts.append(gt)
        all_sq += float(np.sum((act - gt) ** 2))
        n += act.size
    preds, gts = np.array(preds), np.array(gts)
    print(f"traj ({lang[:50]}): T={len(steps)} mse={np.mean((preds-gts)**2):.4f}")
    if not trace_printed:
        trace_printed = True
        for t in range(0, len(steps), 10):
            print(f"  t={t:3d} pred={np.round(preds[t], 2)}")
            print(f"        gt ={np.round(gts[t], 2)}")

print(f"\nOVERALL action MSE: {all_sq / n:.4f}")
print("GT action std per-dim:", np.round(policy.model.dataset_statistics["action"]["std"], 2))
