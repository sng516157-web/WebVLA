# How the Octo policy was trained

Fine-tuning `octo-base-1.5` on LIBERO-Spatial, on a single RTX 4080 Laptop
(12 GB VRAM) in WSL2. Everything below is grounded in the repo:
`setup/finetune_libero_config.py`, `setup/run_finetune.sh`,
`tools/convert_libero_to_rlds.py`, `README.md` (Fine-tuning section).

## Why fine-tune at all

octo-base 1.5 **zero-shot scored 0/6** on the M0 reference eval
(`reference/reference_results.json`). The published LIBERO numbers for Octo
(78.9% on LIBERO-Spatial, OpenVLA paper Table 3) are for the *fine-tuned*
model — Octo ships no official LIBERO eval or checkpoint. So the policy had
to be trained to be a meaningful baseline.

## Data

- **Source:** 10 `libero_spatial` hdf5 files (500 demonstrations) from
  `huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets`.
- **Conversion:** `tools/convert_libero_to_rlds.py` → RLDS/TFDS, with the
  schema of octo's own `debug_dataset` (one episode = list of steps with
  `observation{image, wrist_image, state}`, `action`,
  `language_instruction`, first/last/terminal flags, reward, discount).
- **Conventions (the details that silently break training if wrong):**
  - images rotated 180° (hdf5 stores MuJoCo's upside-down render);
  - images JPEG-encoded uint8, like the Open X-Embodiment datasets;
  - `state` = 8-D concat of ee_pos, ee_ori (axis-angle), gripper_states;
  - actions stored raw (gripper ±1, **not** re-signed);
  - language pulled from the LIBERO benchmark registry, matched via the
    bddl path stored in each hdf5's `env_args`.

## Config

`setup/finetune_libero_config.py` wraps octo's stock `finetune_config.py`
(`full,language_conditioned`) from `hf://rail-berkeley/octo-base-1.5`.
Deviations from the defaults, all hardware-driven and documented in the
config docstring:

| knob | octo default | ours | why |
|---|---|---|---|
| batch size | 256 | 16 | single 12 GB GPU |
| peak LR | 3e-4 | **5e-5** | see the LR finding below |
| warmup steps | 2000 | 500 | shorter run, small batch |
| shuffle buffer | 10000 | 500 | decoded frames OOM-killed the 15 GB WSL VM |
| frame threads | 16 | 6 | same |
| viz eval batch | 128 | 32 | VRAM headroom |

Everything else is octo default: full fine-tune (not LoRA), window_size 1,
action horizon 4, image augmentation, 256² primary / 128² wrist resize,
normal action/proprio normalization with the gripper dim unnormalized
(`action_normalization_mask`).

## The LR finding (the interesting part)

octo's default peak LR 3e-4 is tuned for batch 256. At batch 16 the 10k-step
run **diverged**: action-prediction MSE on held-out demos was 0.67 — worse
than predicting the dataset mean (ground-truth action variance ≈ 0.2) — and
the resulting checkpoint scored 0/6.

Before touching the optimizer, input conventions were verified as not
guilty: image orientation (`setup/check_orientation.py`), proprio
eval-vs-train match (`setup/check_proprio.py`), language mapping, action
statistics. Only then was the LR changed.

5e-5 sits between linear (≈1.9e-5) and square-root (≈7.5e-5) scaling of the
default for the 16× smaller batch. At 5e-5 the held-out action MSE dropped
to **0.079 after only 3k steps**. Diagnostic tool:
`setup/check_action_mse.py <checkpoint_dir>`.

## Run

```bash
bash setup/run_finetune.sh    # ~5 h at ~2.5 it/s, WANDB_MODE=offline
```

50k steps, checkpoints every 5k under `$WEBVLA_CKPTS/octo_finetune/...`,
run name `libero_spatial_bs16_lr5e5`. XLA memory fraction 0.85.

## Validation after training

1. **Reference eval** (closed loop, 2 tasks × 3 seeds): **5/6 = 83.3%**
   (`reference/ft50000/reference_results.json`) — consistent with the
   published 78.9%.
2. **Batch grid** (3 tasks × 5 seeds): **11/15 = 73.3%** (`results.json`);
   failures are grasp-fumble flaky seeds, not systematic errors (video-
   verified).
3. Camera-key sanity check: obs `image_primary` (256,256,3) /
   `image_wrist` (128,128,3) match the checkpoint's `example_batch`
   exactly — recorded in `results.json.sanity_check`.

Eval-side note: the fine-tuned checkpoint's `example_batch` contains
`proprio` and the wrapper feeds normalized proprio, but the sanity check
confirms octo-base does not consume it (image+language only). Consistent
between training and eval, costs nothing.

## Reproduce

```bash
# one-time: dataset download + RLDS conversion
bash setup/run.sh -m tools.convert_libero_to_rlds --suite libero_spatial
# train
bash setup/run_finetune.sh
# check a checkpoint's held-out action MSE
bash setup/run.sh setup/check_action_mse.py $WEBVLA_CKPTS/octo_finetune/...
```
