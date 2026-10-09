"""Validate OctoPolicy loads a locally finetuned checkpoint (CPU-only test).

Checks: checkpoint loads from a local dir, window_size inferred from
example_batch, unnorm_key auto-selected ("action" for finetuned ckpts).
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys

from policy.octo_policy import OctoPolicy

ckpt = sys.argv[1]
policy = OctoPolicy(checkpoint=ckpt)
print("window_size:", policy.window_size)
print("unnorm_key:", policy.unnorm_key)
print("OK")
