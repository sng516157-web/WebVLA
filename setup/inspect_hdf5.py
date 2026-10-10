import glob
import os
import sys

import h5py
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import libero_datasets_dir

path = sorted(glob.glob(os.path.join(libero_datasets_dir("libero_spatial"), "*.hdf5")))[0]
f = h5py.File(path, "r")


def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"DATASET {name} shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"GROUP   {name}")


f.visititems(walk)
print("root attrs:", dict(f.attrs))
print("data attrs keys:", list(f["data"].attrs.keys()))
env_args = f["data"].attrs.get("env_args")
print("env_args:", env_args)
demo0 = f["data/demo_0"]
print("demo_0 keys:", list(demo0.keys()))
print("demo_0 obs keys:", list(demo0["obs"].keys()))
print("actions shape:", demo0["actions"].shape)
