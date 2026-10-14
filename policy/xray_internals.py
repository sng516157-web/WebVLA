"""Read-only internals capture for the LeRobot pi05 policy (X-Ray mode).

READ-ONLY LAW: nothing here may alter model outputs. Capture works by
instance-attribute shadowing (observation-only wrappers around
`denoise_step` / `forward`) and by `output_attentions` config flags that only
materialize tensors the inference path already computes — pi05 inference
forces eager attention (modeling_pi05.py:811,881), so enabling
`output_attentions` does not change the compute path or numerics.

The regression gate that proves this is setup/xray_equivalence_check.py.

Captured internals (all numpy, sim units unless noted):
  - denoise_xt: (10, chunk, 7) — the flow-matching trajectory estimate x_t
    entering each of the 10 Euler steps, unnormalized through the SAME
    official postprocessor the wrapper uses for the final chunk.
  - heatmap_agentview / heatmap_wrist: (256,) float32 in [0,1] — mean
    attention FROM instruction tokens TO image tokens in the prefix LM
    (last 6 layers, mean over heads and over real text tokens), row-major
    16x16 patch grids.
"""

import numpy as np

# Prefix token layout (XRAY_MODEL.md §2, verified against embed_prefix):
N_IMG_TOKENS_PER_VIEW = 256          # 16x16 patches, SigLIP patch14 @224px
PREFIX_AGENTVIEW = (0, 256)
PREFIX_WRIST = (256, 512)
PREFIX_TEXT = (768, 968)             # after the empty-camera pad block
ATTN_LAYERS_KEPT = 6                 # pool the last 6 of 18 prefix layers


class Pi05InternalsCapture:
    """Context manager: observe one predict_action_chunk call.

    want: subset of {"denoise", "attn_prefix"}. Unknown keys are ignored.
    """

    def __init__(self, wrapper, want=("denoise", "attn_prefix"),
                 text_query_mask=None):
        self._wrapper = wrapper            # LerobotPi05Policy
        self._model = wrapper._policy.model  # PI05Pytorch
        self._want = set(want)
        # (200,) bool: which text positions are content words (None = use
        # every non-padding position)
        self._text_query_mask = text_query_mask
        self.denoise_xt_norm = []          # (chunk, 7) normalized, per step
        self._prefix_heat = None           # pooled text->image, (2, 256)
        self._restore = []

    # ------------------------------------------------------------ hooks
    def _patch(self, obj, name, fn):
        orig = getattr(obj, name)          # bound method (class attr)
        self._restore.append((obj, name, orig, "method"))
        object.__setattr__(obj, name, fn)  # instance attr shadows the method
        return orig

    def _patch_config(self, cfg, name, value):
        old = getattr(cfg, name)
        self._restore.append((cfg, name, old, "config"))
        setattr(cfg, name, value)

    def __enter__(self):
        if "denoise" in self._want:
            orig_denoise = None

            def denoise_wrapper(*args, **kwargs):
                # denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)
                x_t = kwargs.get("x_t", args[2] if len(args) > 2 else None)
                if x_t is not None:
                    import torch

                    self.denoise_xt_norm.append(
                        x_t.detach().to(torch.float32).cpu().numpy()[0, :, :7].copy()
                    )
                return orig_denoise(*args, **kwargs)

            orig_denoise = self._patch(self._model, "denoise_step", denoise_wrapper)

        if "attn_prefix" in self._want:
            lm = self._model.paligemma_with_expert.paligemma.language_model
            # sample_actions forces eager at modeling_pi05.py:811 but only
            # DURING the call; the output_attentions setter validates against
            # _attn_implementation, so apply the same forcing first.
            self._patch_config(lm.config, "_attn_implementation", "eager")
            self._patch_config(lm.config, "output_attentions", True)

            def lm_forward_wrapper(*args, **kwargs):
                out = orig_lm_forward(*args, **kwargs)
                self._pool_prefix_attention(getattr(out, "attentions", None),
                                            kwargs.get("attention_mask"))
                return out

            orig_lm_forward = self._patch(lm, "forward", lm_forward_wrapper)
        return self

    def __exit__(self, *exc):
        for obj, name, orig, kind in reversed(self._restore):
            if kind == "config":
                setattr(obj, name, orig)
            else:
                # restore class-method lookup by removing the instance attr
                try:
                    object.__delattr__(obj, name)
                except AttributeError:
                    object.__setattr__(obj, name, orig)
        self._restore.clear()
        return False

    # --------------------------------------------------- attention pooling
    def _pool_prefix_attention(self, attentions, attention_mask):
        """attentions: tuple of 18 tensors (1, heads, 968, 968) bf16."""
        if attentions is None:
            return
        import torch

        n = len(attentions)
        layers = attentions[max(0, n - ATTN_LAYERS_KEPT):]
        # Which text positions are real (not right-padding)? The additive 4D
        # attention mask marks padded columns -inf for every query row.
        text_lo, text_hi = PREFIX_TEXT
        valid_text = None
        if attention_mask is not None:
            m = attention_mask
            if m.dim() == 4:               # (B, 1, q, k) additive
                col_ok = (m[0, 0].amax(dim=0) > -1e4)  # (k,)
            elif m.dim() == 2:             # (B, k) bool pad mask
                col_ok = m[0].bool()
            else:
                col_ok = None
            if col_ok is not None and col_ok.shape[0] >= text_hi:
                valid_text = col_ok[text_lo:text_hi]

        acc = torch.zeros(2, N_IMG_TOKENS_PER_VIEW, dtype=torch.float32)
        qmask = self._text_query_mask
        if qmask is not None and valid_text is not None:
            qmask = [q and bool(v) for q, v in zip(qmask, valid_text.tolist())]
        for attn in layers:                # (1, heads, 968, 968)
            a = attn[0].to(torch.float32)  # (heads, q, k)
            q_text = a[:, text_lo:text_hi, :]
            if qmask is not None:
                sel = torch.tensor(qmask, dtype=torch.bool,
                                   device=q_text.device)
                if sel.any():
                    q_text = q_text[:, sel, :]
            elif valid_text is not None:
                q_text = q_text[:, valid_text.to(q_text.device), :]
            # mean over heads and real text queries -> (968,) key profile
            key_profile = q_text.mean(dim=(0, 1))
            acc[0] += key_profile[PREFIX_AGENTVIEW[0]:PREFIX_AGENTVIEW[1]].cpu()
            acc[1] += key_profile[PREFIX_WRIST[0]:PREFIX_WRIST[1]].cpu()
        acc /= max(1, len(layers))
        self._prefix_heat = acc

    # ------------------------------------------------------------- results
    @staticmethod
    def _normalize01(v):
        """Percentile-clipped scaling: SigLIP/Gemma attention has strong
        border-patch sinks; raw min-max lets one sink cell flatten all
        object-level structure. Clip at p2/p98 so the map shows the
        distribution over the scene, not the sink. Labeled as such in the UI."""
        v = np.asarray(v, dtype=np.float32)
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        if hi - lo < 1e-8:
            return np.zeros_like(v, dtype=np.float32)
        return np.clip((v - lo) / (hi - lo), 0.0, 1.0)

    def denoise_trajectory(self):
        """(n_steps, chunk, 7), unnormalized via the official postprocessor."""
        if not self.denoise_xt_norm:
            return None
        import torch

        out = []
        for x in self.denoise_xt_norm:     # (chunk, 7) normalized
            t = torch.from_numpy(x).unsqueeze(0)  # (1, chunk, 7) — same shape act() uses
            out.append(np.asarray(
                self._wrapper._postprocess(t)[0], dtype=np.float32))
        return np.stack(out)

    def heatmaps(self):
        if self._prefix_heat is None:
            return {}
        return {
            "heatmap_agentview": self._normalize01(self._prefix_heat[0].numpy()),
            "heatmap_wrist": self._normalize01(self._prefix_heat[1].numpy()),
        }
