"""Thin wrapper around LIBERO OffScreenRenderEnv producing Octo-ready observations.

Conventions follow the de-facto standard LIBERO eval harness
(openvla/openvla experiments/robot/libero), which is the environment side of the
published LIBERO numbers for VLA policies:
  - OffScreenRenderEnv created from the task's BDDL file at render_size x render_size
  - env.seed() called once at env creation (LIBERO procedural layout RNG)
  - initial object poses pinned via benchmark.get_task_init_states(task_id)[seed]
  - `settle_steps` no-op steps after reset (sim drops objects; let them fall)
  - images rotated 180 degrees to match training preprocessing
  - resize via JPEG round-trip + lanczos3 (matches the Octo dataloader)

NOTE: the octo-models/octo repo ships no LIBERO eval script (checked main,
main_old, all tags). The policy-side mechanics (sample_actions with
unnormalization_statistics, history window, receding-horizon execution) follow
octo's official examples/03_eval_finetuned.py.
"""

import math
import os

import numpy as np

# No-op action: zero arm motion, gripper open (-1).
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

# Octo camera keys -> LIBERO/robosuite observation keys.
SIM_CAMERA_KEYS = {
    "image_primary": "agentview_image",
    "image_wrist": "robot0_eye_in_hand_image",
}

# Resolutions octo-base was trained with (per HF model card).
POLICY_IMAGE_SIZE = {
    "image_primary": (256, 256),
    "image_wrist": (128, 128),
}

# Episode step limits per suite, from the standard LIBERO eval harness
# (comment there: longest training demo per suite).
MAX_STEPS_PER_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def quat2axisangle(quat):
    """Convert quaternion (x,y,z,w) to axis-angle. Verbatim from robosuite
    transform_utils (as copied in openvla libero_utils)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def resize_image(img, size):
    """JPEG round-trip + lanczos3 resize, matching the Octo dataloader."""
    import tensorflow as tf

    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def _rotate180(img):
    return img[::-1, ::-1]


class LiberoWrapper:
    """reset(task_id, seed) / step(action) / render(camera_name) / obs()."""

    def __init__(self, suite_name="libero_spatial", render_size=256, settle_steps=10, env_seed=0):
        from libero.libero import benchmark

        self.suite_name = suite_name
        self.benchmark = benchmark.get_benchmark_dict()[suite_name]()
        self.render_size = render_size
        self.settle_steps = settle_steps
        self.env_seed = env_seed  # layout RNG seed, fixed at env creation (harness convention)
        self.max_steps = MAX_STEPS_PER_SUITE[suite_name]

        self.env = None
        self.task_id = None
        self.task = None
        self.instruction = None
        self.init_state_idx = None
        self._raw = None
        self.steps = 0

    def _make_env(self, task_id):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        if self.env is not None:
            self.env.close()
        self.task = self.benchmark.get_task(task_id)
        self.instruction = self.task.language
        bddl_file = os.path.join(
            get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file
        )
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=self.render_size,
            camera_widths=self.render_size,
        )
        # IMPORTANT (harness convention): seed affects object positions even when
        # using fixed initial states, so pin it once at creation.
        self.env.seed(self.env_seed)
        self.task_id = task_id

    def reset(self, task_id, seed):
        if self.env is None or self.task_id != task_id:
            self._make_env(task_id)
        init_states = self.benchmark.get_task_init_states(task_id)
        self.init_state_idx = seed % len(init_states)
        self.env.reset()
        raw = self.env.set_init_state(init_states[self.init_state_idx])
        # Let dropped objects settle before the policy takes over.
        for _ in range(self.settle_steps):
            raw, _, _, _ = self.env.step(DUMMY_ACTION)
        self._raw = raw
        self.steps = 0
        return self.obs()

    def step(self, action):
        raw, reward, done, info = self.env.step(np.asarray(action, dtype=np.float64))
        self._raw = raw
        self.steps += 1
        return self.obs(), reward, done, info

    def render(self, camera_name="agentview"):
        """HWC uint8, rotated 180 to be upright (same orientation the policy sees)."""
        key = SIM_CAMERA_KEYS.get(camera_name, camera_name)
        return _rotate180(self._raw[key]).copy()

    def obs(self):
        raw = self._raw
        return {
            "image_primary": resize_image(_rotate180(raw["agentview_image"]), POLICY_IMAGE_SIZE["image_primary"]),
            "image_wrist": resize_image(
                _rotate180(raw["robot0_eye_in_hand_image"]), POLICY_IMAGE_SIZE["image_wrist"]
            ),
            "proprio": np.concatenate(
                (
                    raw["robot0_eef_pos"],
                    quat2axisangle(raw["robot0_eef_quat"]),
                    raw["robot0_gripper_qpos"],
                )
            ).astype(np.float32),
            "instruction": self.instruction,
        }

    def check_success(self):
        return bool(self.env.check_success())

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
