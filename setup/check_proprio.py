"""Compare eval-time proprio (sim wrapper) vs training-time proprio (RLDS state)
at episode start, plus normalization stats, to detect convention mismatch."""
import json

import numpy as np

from sim.wrapper import LiberoWrapper

env = LiberoWrapper("libero_spatial")
obs = env.reset(task_id=0, seed=0)
print("eval proprio @reset:  ", np.round(obs["proprio"], 4))

# training state at demo start (same task's hdf5, demo_0)
import h5py

H5 = "/root/webvla/LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
with h5py.File(H5, "r") as f:
    st = f["data/demo_0/obs/ee_pos"][0].tolist() + \
         f["data/demo_0/obs/ee_ori"][0].tolist() + \
         f["data/demo_0/obs/gripper_states"][0].tolist()
print("train state @demo0 t0:", np.round(st, 4))

stats = json.load(open("/root/webvla/octo_ckpts/octo_finetune/libero_spatial/"
                       "libero_spatial_bs16_20261010_023740/dataset_statistics.json"))
print("stats keys:", list(stats.keys()))
print("proprio mean:", np.round(stats["proprio"]["mean"], 3))
print("proprio std: ", np.round(stats["proprio"]["std"], 3))
print("action mean:", np.round(stats["action"]["mean"], 3))
print("action std: ", np.round(stats["action"]["std"], 3))
