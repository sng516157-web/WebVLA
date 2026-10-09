"""Octo policy wrapper: checkpoint loading, task creation, windowed chunk prediction.

Policy-side mechanics mirror octo's official examples/03_eval_finetuned.py:
    policy_fn = supply_rng(partial(model.sample_actions,
                                   unnormalization_statistics=model.dataset_statistics[...]["action"]))
plus octo.utils.gym_wrappers.stack_and_pad for the history window (verbatim reuse).

De-normalization is delegated entirely to model.sample_actions
(unnormalization_statistics -> mean/std unnormalize with mask; mask leaves the
gripper dim un-normalized, matching training). Nothing is improvised here.
"""

from collections import deque
from functools import partial

import numpy as np

DEFAULT_CHECKPOINT = "hf://rail-berkeley/octo-base-1.5"

# Preference order for action de-normalization statistics of the *pretrained*
# octo-base checkpoint on the LIBERO (Franka Panda) embodiment: Franka-arm OXE
# datasets first. Fine-tuned checkpoints carry a top-level "action" entry, which
# always wins.
UNNORM_KEY_PREFERENCE = [
    "taco_play",
    "taco_extra",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "berkeley_cable_routing",
]


class OctoPolicy:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, window_size=2, unnorm_key=None):
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")  # reserve the GPU for JAX

        import jax
        from octo.model.octo_model import OctoModel
        from octo.utils.train_callbacks import supply_rng

        self.checkpoint = checkpoint
        self.window_size = window_size
        self.model = OctoModel.load_pretrained(checkpoint)

        stats = self.model.dataset_statistics
        if unnorm_key is not None:
            self.unnorm_key = unnorm_key
        elif "action" in stats:  # fine-tuned checkpoint
            self.unnorm_key = "action"
        else:
            for k in UNNORM_KEY_PREFERENCE:
                if k in stats:
                    self.unnorm_key = k
                    break
            else:
                raise KeyError(
                    f"No suitable action statistics key. Available: {sorted(stats.keys())}"
                )
        unnorm_stats = stats[self.unnorm_key]["action"] if self.unnorm_key != "action" else stats["action"]

        # octo's official eval recipe (03_eval_finetuned.py)
        self._policy_fn = supply_rng(
            partial(self.model.sample_actions, unnormalization_statistics=unnorm_stats)
        )
        self._jax = jax

        self._history = deque(maxlen=window_size)
        self._num_obs = 0
        self._task = None

    @property
    def expected_observation(self):
        """Observation keys/shapes the checkpoint was trained with (from example_batch)."""
        import jax

        return jax.tree_map(lambda x: x.shape, self.model.example_batch["observation"])

    def reset(self, instruction):
        self._history.clear()
        self._num_obs = 0
        self._task = self.model.create_tasks(texts=[instruction])

    def act(self, obs):
        """One observation dict -> action chunk of shape (action_horizon, 7)."""
        from octo.utils.gym_wrappers import stack_and_pad

        obs = {k: v for k, v in obs.items() if isinstance(v, np.ndarray)}
        if self._num_obs == 0:
            # HistoryWrapper.reset semantics: pad with copies of the first obs.
            self._history.extend([obs] * self.window_size)
            self._num_obs = 1
        else:
            self._history.append(obs)
            self._num_obs += 1
        full_obs = stack_and_pad(self._history, self._num_obs)
        batched = self._jax.tree_map(lambda x: x[None], full_obs)
        actions = self._policy_fn(batched, self._task)
        return np.asarray(actions[0], dtype=np.float32)
