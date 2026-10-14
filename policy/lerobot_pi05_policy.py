"""pi0.5 via LeRobot's PyTorch port (lerobot PI05Policy) — no JAX, no policy server.

Duck-typed drop-in for policy.pi0_policy.Pi05Policy (same OBS_METHOD, reset /
set_instruction / act / expected_observation), so runner/, server/ and live/
consume it unchanged. It consumes the same `LiberoWrapper.obs_pi05()` dict and
returns the first REPLAN_STEPS (5) actions of the model's chunk.

Input conventions mirror LeRobot's own LIBERO eval (lerobot-eval --env.type=libero)
exactly, as of lerobot 0.4.4:

  - Keys: the checkpoint (lerobot/pi05_libero_finetuned_v044, config.json
    input_features) expects "observation.images.image" (agentview),
    "observation.images.image2" (wrist), "observation.state" (8-D),
    plus the language instruction under "task".
  - Images: LeRobot's pipeline is env raw frame -> [0,1] float CHW
    (lerobot/envs/utils.py: preprocess_observation) -> 180-degree rotation
    (lerobot/processor/env_processor.py: LiberoProcessorStep, torch.flip
    dims=[2,3]). obs_pi05() already returns 180-rotated uint8 frames (openpi
    convention), so act() converts to float [0,1] CHW and deliberately does
    NOT flip again — the end result matches LeRobot's pipeline bit-for-bit.
  - State: LiberoProcessorStep builds [eef_pos(3), quat2axisangle(eef_quat)(3),
    gripper_qpos(2)] float32 — identical layout to obs_pi05()'s
    "observation/state", so it is passed through (cast to float32).
  - Language: the checkpoint's preprocessor (policy_preprocessor.json) runs
    Pi05PrepareStateTokenizerProcessorStep, which discretizes the NORMALIZED
    state into 256 bins and embeds it in the prompt
    ("Task: ..., State: <bins>;\nAction: ") exactly like openpi's
    PaligemmaTokenizer, then tokenizes with the PaliGemma tokenizer at
    max_length=200, right padding.

Tokenizer note: the checkpoint's tokenizer_processor points at
google/paligemma-3b-pt-224, which is license-GATED on HF (401 without an
accepted-license token). We instead load the same sentencepiece vocab from a
local file (shipped with openpi's assets cache) via GemmaTokenizer; token ids
verified identical to openpi's encode(..., add_bos=True). Override with the
WEBVLA_PALIGEMMA_TOKENIZER env var or the tokenizer_path argument. On a host
with an HF token that accepted the license (e.g. a Space secret), the gated
name also works as a fallback.

Checkpoint config overrides applied at load: device (default cpu) and
compile_model=False — the checkpoint requests torch.compile(mode=
"max-autotune"), which costs minutes of compile time per process and is not
worth it for single-env CPU/GPU eval.
"""

import os

import numpy as np

# Must run before the first `import transformers` (see module docstring):
# on the ZeroGPU image, gradio 6 forces huggingface-hub>=1.16 while the
# lerobot transformers fork declares <1.0; this relaxes the version gate.
from . import _hub_compat

_hub_compat.apply()

DEFAULT_MODEL_ID = "lerobot/pi05_libero_finetuned_v044"

# openpi recipe parity with the JAX Pi05Policy: execute 5 of the chunk.
REPLAN_STEPS = 5
IMAGE_SIZE = (224, 224)

# Local copies of the PaliGemma sentencepiece vocab (google/paligemma-3b-pt-224
# is license-gated on HF). First match wins.
_TOKENIZER_CANDIDATES = (
    os.environ.get("WEBVLA_PALIGEMMA_TOKENIZER"),
    "/root/.cache/openpi/big_vision/paligemma_tokenizer.model",
    os.path.expanduser("~/.cache/openpi/big_vision/paligemma_tokenizer.model"),
)


def _load_paligemma_tokenizer(tokenizer_path=None):
    from transformers import GemmaTokenizer

    candidates = [tokenizer_path, *_TOKENIZER_CANDIDATES]
    for path in candidates:
        if path and os.path.exists(path):
            print(f"[lerobot-pi05] tokenizer: local sentencepiece {path}", flush=True)
            return GemmaTokenizer(vocab_file=path)
    # Fall back to the gated HF name (works with an accepted-license HF token).
    from transformers import AutoTokenizer

    print("[lerobot-pi05] tokenizer: HF google/paligemma-3b-pt-224 (needs HF token)", flush=True)
    return AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")


class LerobotPi05Policy:
    OBS_METHOD = "obs_pi05"  # LiberoWrapper method the runner should use

    def __init__(self, model_id=DEFAULT_MODEL_ID, device=None, tokenizer_path=None,
                 replan_steps=REPLAN_STEPS):
        import torch  # noqa: F401  (defer heavy imports to construction)
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05 import PI05Policy

        self.checkpoint = model_id  # run_grid records this as metadata
        self.replan_steps = replan_steps
        self.device = device or os.environ.get("WEBVLA_LEROBOT_DEVICE", "cpu")

        cfg = PreTrainedConfig.from_pretrained(model_id)
        cfg.device = self.device
        cfg.compile_model = False  # skip max-autotune compile (minutes per process)
        cfg.gradient_checkpointing = False  # inference only
        self._policy = PI05Policy.from_pretrained(model_id, config=cfg)
        self._policy.eval()

        tokenizer = _load_paligemma_tokenizer(tokenizer_path)
        self._tokenizer = tokenizer  # X-Ray: decode text-token positions
        self._preprocess, self._postprocess = make_pre_post_processors(
            cfg,
            model_id,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
                "tokenizer_processor": {"tokenizer": tokenizer},
            },
        )
        self._instruction = None

    @property
    def expected_observation(self):
        """Keys/shapes the policy consumes (for runner.batch.sanity_check)."""
        return {
            "observation/image": (*IMAGE_SIZE, 3),
            "observation/wrist_image": (*IMAGE_SIZE, 3),
            "observation/state": (8,),
        }

    def reset(self, instruction):
        self._instruction = instruction
        self._policy.reset()  # clear LeRobot's internal action queue

    def set_instruction(self, instruction):
        """M2 parity: swap language conditioning mid-episode."""
        self._instruction = instruction
        self._policy.reset()

    @staticmethod
    def _img_to_tensor(img):
        """(H, W, 3) uint8, already 180-rotated -> (1, 3, H, W) float32 [0, 1].

        Equivalent to LeRobot's preprocess_observation + LiberoProcessorStep
        applied to the raw (unrotated) sim frame.
        """
        import torch

        t = torch.from_numpy(np.ascontiguousarray(img))
        return t.permute(2, 0, 1).unsqueeze(0).to(torch.float32).div_(255.0)

    def _build_batch(self, obs):
        import torch

        state = np.asarray(obs["observation/state"], dtype=np.float32)
        assert state.shape == (8,), f"expected 8-D state, got {state.shape}"
        return {
            "observation.images.image": self._img_to_tensor(obs["observation/image"]),
            "observation.images.image2": self._img_to_tensor(obs["observation/wrist_image"]),
            "observation.state": torch.from_numpy(state).unsqueeze(0),
            "task": [str(self._instruction)],
        }

    def act(self, obs):
        """obs (pi05 format, see LiberoWrapper.obs_pi05) -> (replan_steps, 7)."""
        import torch

        batch = self._preprocess(self._build_batch(obs))
        with torch.inference_mode():
            chunk = self._policy.predict_action_chunk(batch)  # (1, chunk_size, 7), normalized
        chunk = self._postprocess(chunk)  # unnormalized, on CPU
        chunk = np.asarray(chunk[0], dtype=np.float32)
        assert len(chunk) >= self.replan_steps, (
            f"policy returned {len(chunk)} actions, need {self.replan_steps}"
        )
        return chunk[: self.replan_steps]

    def act_with_internals(self, obs, want=("denoise", "attn_prefix")):
        """act()'s exact code path, observed by Pi05InternalsCapture.

        Returns (exec_chunk, full_chunk, internals):
          exec_chunk  (replan_steps, 7) — identical to what act() returns
          full_chunk  (chunk_size, 7)   — the whole predicted trajectory
          internals   dict with "denoise_xt" (n_steps, chunk_size, 7) and
                      "heatmap_agentview"/"heatmap_wrist" (256,) per `want`
        """
        import re

        import torch

        from .xray_internals import Pi05InternalsCapture

        batch = self._preprocess(self._build_batch(obs))

        # Content-word query mask for the attention heatmap: averaging over
        # ALL text tokens dilutes grounding with scaffold/padding sinks, so
        # pool attention only from real alphabetic instruction tokens.
        # Text tokens sit at prefix positions 768+i, right-padded (XRAY_MODEL.md §2).
        text_query_mask = None
        tokens = batch.get("observation.language.tokens")
        tmask = batch.get("observation.language.attention_mask")
        if tokens is not None:
            ids = tokens[0].tolist()
            valid = (tmask[0].tolist() if tmask is not None
                     else [1] * len(ids))
            pieces = self._tokenizer.convert_ids_to_tokens(ids)
            text_query_mask = [
                bool(v) and bool(re.fullmatch(r"▁?[A-Za-z]{2,}", p))
                for v, p in zip(valid, pieces)
            ]

        with Pi05InternalsCapture(self, want,
                                  text_query_mask=text_query_mask) as cap:
            with torch.inference_mode():
                chunk = self._policy.predict_action_chunk(batch)
        chunk = self._postprocess(chunk)
        full = np.asarray(chunk[0], dtype=np.float32)
        assert len(full) >= self.replan_steps, (
            f"policy returned {len(full)} actions, need {self.replan_steps}"
        )
        internals = {"chunk_size": int(full.shape[0])}
        if "denoise" in want:
            traj = cap.denoise_trajectory()
            if traj is not None:
                internals["denoise_xt"] = traj
        internals.update(cap.heatmaps())
        return full[: self.replan_steps], full, internals
