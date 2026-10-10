"""Compare eval-time proprio (sim wrapper) vs training-time proprio (RLDS state)
at episode start, plus normalization stats, to detect convention mismatch.

Usage: bash setup/run.sh setup/check_proprio.py [checkpoint_dir]
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import CKPTS_DIR, libero_datasets_dir

from sim.wrapper import LiberoWrapper

env = LiberoWrapper("libero_spatial")
obs = env.reset(task_id=0, seed=0)
print("eval proprio @reset:  ", np.round(obs["proprio"], 4))

# training state at demo start (first libero_spatial hdf5, demo_0)
import h5py

H5 = sorted(glob.glob(os.path.join(libero_datasets_dir("libero_spatial"), "*.hdf5")))[0]
with h5py.File(H5, "r") as f:
    st = f["data/demo_0/obs/ee_pos"][0].tolist() + \
         f["data/demo_0/obs/ee_ori"][0].tolist() + \
         f["data/demo_0/obs/gripper_states"][0].tolist()
print("train state @demo0 t0:", np.round(st, 4))

ckpt = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WEBVLA_FT_CKPT", CKPTS_DIR)
stats = json.load(open(os.path.join(ckpt, "dataset_statistics.json")))
print("stats keys:", list(stats.keys()))
print("proprio mean:", np.round(stats["proprio"]["mean"], 3))
print("proprio std: ", np.round(stats["proprio"]["std"], 3))
print("action mean:", np.round(stats["action"]["mean"], 3))
print("action std: ", np.round(stats["action"]["std"], 3))
