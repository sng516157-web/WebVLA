"""Convert LIBERO hdf5 demos to an RLDS/TFDS dataset for octo's finetune.py.

Schema matches octo's tests/debug_dataset (bridge_dataset):
  episode = {"steps": [{observation{image,wrist_image,state}, action,
                        language_instruction, is_first, is_last, is_terminal,
                        reward, discount}], "episode_metadata": {"file_path": ...}}

Conventions (matching how LIBERO demos are consumed by VLA training pipelines):
  - images rotated 180 degrees (hdf5 stores mujoco's upside-down render)
  - images JPEG-encoded uint8 (tfds Image feature, like OXE datasets)
  - state = concat(ee_pos, ee_ori (axis-angle as stored), gripper_states)  (8-dim)
  - actions stored raw (env convention; gripper +-1, NOT re-signed)
  - language from the LIBERO benchmark registry, matched via the bddl path
    stored in the hdf5 env_args (fallback: filename heuristic)

Usage (WSL):
  bash setup/run.sh -m tools.convert_libero_to_rlds --suite libero_spatial
Output: $WEBVLA_RLDS/libero_rlds/1.0.0/...   (see paths.py)
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RLDS_DIR, libero_datasets_dir


def build_language_map(suite_name):
    """bddl basename -> task language, from the LIBERO benchmark registry."""
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    m = {}
    for i in range(suite.get_num_tasks()):
        task = suite.get_task(i)
        m[os.path.basename(task.bddl_file)] = task.language
    return m


def language_for_file(hdf5_path, lang_map):
    with h5py.File(hdf5_path, "r") as f:
        env_args = f["data"].attrs.get("env_args", None)
    if env_args:
        try:
            ea = json.loads(env_args)
            bddl = os.path.basename(ea.get("bddl_file_name", "") or ea.get("bddl_file", ""))
            if bddl in lang_map:
                return lang_map[bddl]
        except Exception:
            pass
    # fallback: filename minus scene prefix and suffix
    stem = os.path.basename(hdf5_path).replace("_demo.hdf5", "")
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if p.upper().startswith("SCENE"):
            return " ".join(parts[i + 1 :]).lower()
    return stem.replace("_", " ").lower()


def make_builder(suite_name):
    import tensorflow as tf
    import tensorflow_datasets as tfds

    class LiberoRlds(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "LIBERO hdf5 -> RLDS for octo finetuning."}

        def _info(self):
            return self.dataset_info_from_configs(
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": tfds.features.FeaturesDict(
                                    {
                                        "image": tfds.features.Image(
                                            shape=(128, 128, 3), encoding_format="jpeg"
                                        ),
                                        "wrist_image": tfds.features.Image(
                                            shape=(128, 128, 3), encoding_format="jpeg"
                                        ),
                                        "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                                    }
                                ),
                                "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                "language_instruction": tfds.features.Text(),
                                "is_first": tfds.features.Scalar(dtype=np.bool_),
                                "is_last": tfds.features.Scalar(dtype=np.bool_),
                                "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                                "reward": tfds.features.Scalar(dtype=np.float32),
                                "discount": tfds.features.Scalar(dtype=np.float32),
                            }
                        ),
                        "episode_metadata": tfds.features.FeaturesDict(
                            {"file_path": tfds.features.Text()}
                        ),
                    }
                )
            )

        def _split_generators(self, dl_manager):
            files = sorted(
                glob.glob(os.path.join(libero_datasets_dir(suite_name), "*.hdf5"))
            )
            assert files, f"no hdf5 files found for suite {suite_name}"
            return {"train": self._generate_examples(files), "val": self._generate_examples(files)}

        def _generate_examples(self, files):
            lang_map = build_language_map(suite_name)
            for path in files:
                language = language_for_file(path, lang_map)
                with h5py.File(path, "r") as f:
                    demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
                    for demo_key in demos:
                        demo = f["data"][demo_key]
                        actions = demo["actions"][()].astype(np.float32)
                        T = actions.shape[0]
                        img = demo["obs/agentview_rgb"][()]
                        wrist = demo["obs/eye_in_hand_rgb"][()]
                        state = np.concatenate(
                            [
                                demo["obs/ee_pos"][()],
                                demo["obs/ee_ori"][()],
                                demo["obs/gripper_states"][()],
                            ],
                            axis=-1,
                        ).astype(np.float32)
                        steps = []
                        for t in range(T):
                            steps.append(
                                {
                                    "observation": {
                                        # rotate 180: hdf5 stores mujoco's flipped render
                                        "image": img[t][::-1, ::-1],
                                        "wrist_image": wrist[t][::-1, ::-1],
                                        "state": state[t],
                                    },
                                    "action": actions[t],
                                    "language_instruction": language,
                                    "is_first": t == 0,
                                    "is_last": t == T - 1,
                                    "is_terminal": t == T - 1,
                                    "reward": float(t == T - 1),
                                    "discount": 1.0,
                                }
                            )
                        yield f"{os.path.basename(path)}__{demo_key}", {
                            "steps": steps,
                            "episode_metadata": {"file_path": path},
                        }

    return LiberoRlds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--out", default=RLDS_DIR)
    args = p.parse_args()

    builder_cls = make_builder(args.suite)
    builder = builder_cls(data_dir=args.out)
    builder.download_and_prepare()
    print("WROTE:", builder.data_path)
    # quick readback
    ds = builder.as_dataset(split="train")
    n = sum(1 for _ in ds)
    print("train episodes:", n)


if __name__ == "__main__":
    main()
