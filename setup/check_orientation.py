"""Extract one frame from (a) raw LIBERO hdf5 and (b) converted RLDS dataset,
save side by side as PNGs for visual orientation check."""
import h5py
import numpy as np
import cv2

H5 = "/root/webvla/LIBERO/libero/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"

with h5py.File(H5, "r") as f:
    img_h5 = f["data/demo_0/obs/agentview_rgb"][0]

cv2.imwrite("/tmp/frame_hdf5_raw.png", cv2.cvtColor(img_h5, cv2.COLOR_RGB2BGR))

import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], "GPU")

ds = tfds.load("libero_rlds", data_dir="/root/webvla/rlds_datasets", split="train")
for ep in ds.take(1):
    step0 = next(iter(ep["steps"]))
    img_rlds = step0["observation"]["image"].numpy()
    cv2.imwrite("/tmp/frame_rlds.png", cv2.cvtColor(img_rlds, cv2.COLOR_RGB2BGR))
    print("rlds image shape:", img_rlds.shape)
    print("language:", step0["language_instruction"].numpy())
    print("action[0]:", step0["action"].numpy())
    print("state[0]:", step0["observation"]["state"].numpy())
print("hdf5 image shape:", img_h5.shape)
print("WROTE /tmp/frame_hdf5_raw.png /tmp/frame_rlds.png")
