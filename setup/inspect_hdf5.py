import h5py
import json
import numpy as np

path = "/root/webvla/LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
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
