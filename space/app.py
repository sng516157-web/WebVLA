"""WebVLA π0.5 — Gradio SDK Space on ZeroGPU.

Architecture (ZeroGPU is Gradio-SDK-only and PyTorch-only):
  - Main process (CPU): LIBERO sim (mujoco/robosuite, OSMesa), episode loop,
    frame rendering, mp4 writing, Gradio UI. Gradio's own queue is the only
    queue (default_concurrency_limit=1 -> one episode at a time, visitors see
    their queue position).
  - GPU workers: policy inference only, one @spaces.GPU call per action chunk
    (5 sim steps). The LeRobot PyTorch π0.5 model is loaded at module scope;
    ZeroGPU packs the weights at startup and streams them into a forked GPU
    worker per call. Numpy in/out only (args cross via pickle).

The policy adapter is the same policy/lerobot_pi05_policy.py validated
locally against the JAX π0.5 reference (see README/MIGRATION.md).

Quota: every visitor gets a daily ZeroGPU budget (anon ~2 min, free login
~5 min, PRO ~40 min). A chunk call is ~1-3 s of GPU; an episode is ~10-45
chunks. Quota-exhausted raises inside the GPU call — we catch it, mark the
episode failed, and show the reset time instead of a stack trace.
"""

import json
import os
import time
import traceback
from datetime import datetime, timezone

import numpy as np

# --- rendering backend BEFORE any mujoco/robosuite import ---
from server.bootstrap_osmesa import ensure as _osmesa_ensure

_osmesa_ensure()

# --- LIBERO source (pip drops its data files) + lerobot (pip deps conflict
# with the sim stack; install --no-deps at boot) — both no-ops locally ---
try:
    from bootstrap_libero import ensure as _libero_ensure
except ImportError:  # running from the repo checkout (space/ subdir)
    from space.bootstrap_libero import ensure as _libero_ensure

_libero_ensure()


def _fork_transformers_installed():
    """True only if transformers is the lerobot fork (direct_url marker)."""
    import glob as _glob
    import json as _json
    import importlib.util as _ilu

    spec = _ilu.find_spec("transformers")
    if spec is None or not spec.origin:
        return False
    site = os.path.dirname(os.path.dirname(spec.origin))
    for di in _glob.glob(os.path.join(site, "transformers-*.dist-info")):
        du = os.path.join(di, "direct_url.json")
        if os.path.exists(du):
            try:
                with open(du) as f:
                    d = _json.load(f)
                return (d.get("vcs_info", {}).get("requested_revision")
                        == "fix/lerobot_openpi")
            except Exception:
                return False
    return False


def _ensure_lerobot():
    # Both --no-deps: lerobot's pip deps force numpy>=2 (breaks gym 0.25.2),
    # and the transformers fork declares huggingface-hub<1.0 (conflicts with
    # gradio 6.30's hub>=1.16 at pip-resolution time). All real deps are in
    # requirements.txt; policy/_hub_compat.py relaxes the fork's runtime
    # version gate before the first transformers import.
    try:
        import lerobot  # noqa: F401
    except ImportError:
        import subprocess
        import sys
        print("[space] installing lerobot==0.4.4 (--no-deps) ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                        "--quiet", "lerobot==0.4.4"], check=True)
    if not _fork_transformers_installed():
        import subprocess
        import sys
        print("[space] installing transformers fork (--no-deps) ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                        "--quiet", "--force-reinstall",
                        "git+https://github.com/huggingface/transformers.git"
                        "@fix/lerobot_openpi"], check=True)


_ensure_lerobot()

import gradio as gr

try:
    import spaces
except ImportError:  # local dev without the spaces package
    class _SpacesShim:
        @staticmethod
        def GPU(duration=None):
            def deco(fn):
                return fn
            return deco
    spaces = _SpacesShim()

import paths
from sim.wrapper import LiberoWrapper

SUITE = "libero_spatial"
EPISODES_DIR = os.path.join(paths.REPO_ROOT, "episodes")
SCENES_DIR = os.path.join(paths.REPO_ROOT, "assets", "scenes")
os.makedirs(EPISODES_DIR, exist_ok=True)

# The PaliGemma tokenizer is license-gated on HF; we ship the sentencepiece
# vocab in the Space repo (identical ids, verified against openpi).
_TOK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "paligemma_tokenizer.model")
if os.path.exists(_TOK):
    os.environ.setdefault("WEBVLA_PALIGEMMA_TOKENIZER", _TOK)

# ---------------------------------------------------------------- model
# Module-scope load: on ZeroGPU the `.to("cuda")` inside from_pretrained is
# intercepted, weights are packed to disk, and each @spaces.GPU call runs in
# a forked worker with the weights on VRAM (kept warm between calls).
# Locally (no ZeroGPU runtime) the model simply sits on the local GPU/CPU.
_DEVICE = "cuda" if os.environ.get("SPACE_ID") else os.environ.get(
    "WEBVLA_LEROBOT_DEVICE", "cpu")

print(f"[space] loading LeRobot pi05 on {_DEVICE} ...", flush=True)
_t0 = time.time()
from policy.lerobot_pi05_policy import LerobotPi05Policy

policy = LerobotPi05Policy(device=_DEVICE)
print(f"[space] policy loaded in {time.time()-_t0:.1f}s", flush=True)

print("[space] building LIBERO env ...", flush=True)
env = LiberoWrapper(SUITE, render_size=256)
TASKS = [(f"task {i} — {env.benchmark.get_task(i).language}", i)
         for i in range(env.benchmark.get_num_tasks())]
print(f"[space] {len(TASKS)} tasks ready", flush=True)


@spaces.GPU(duration=15)
def infer_chunk(img, wrist, state, instruction):
    """One action chunk on the GPU worker. Numpy in, numpy out."""
    obs = {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
    }
    policy.reset(instruction)
    return policy.act(obs)


# ---------------------------------------------------------------- episode
def _save_episode(task_id, seed, frames, meta):
    import imageio.v2 as imageio

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{SUITE}__task{task_id}__seed{seed}__{ts}"
    mp4 = os.path.join(EPISODES_DIR, base + ".mp4")
    writer = imageio.get_writer(mp4, fps=30)
    for f in frames:
        writer.append_data(f)
    writer.close()
    meta["mp4"] = base + ".mp4"
    with open(os.path.join(EPISODES_DIR, base + ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return mp4


def run_episode(task_id, seed, custom_instruction, progress=gr.Progress()):
    """Gradio generator: streams progress, ends with the recorded video."""
    empty = (None, None, "", None)
    if task_id is None:
        yield (*empty[:3], "Pick a task first.")
        return
    task_id, seed = int(task_id), int(seed or 0)
    custom = (custom_instruction or "").strip() or None

    progress(0, desc="resetting scene ...")
    obs = env.reset(task_id, seed)
    instruction = custom or obs["instruction"]
    if custom:
        env.instruction = custom
    base_language = env.benchmark.get_task(task_id).language

    meta = {
        "suite": SUITE, "task_id": task_id, "seed": seed,
        "instruction": instruction, "policy_backend": "lerobot_pi05",
        "checkpoint": policy.checkpoint,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "exec_horizon": policy.replan_steps, "fps": 30,
    }
    if custom:
        meta["instruction_overridden"] = True
        meta["base_task_language"] = base_language

    frames = [env.render("agentview")]
    latencies = []
    done = False
    t_start = time.time()
    # env override caps episode length (smoke tests; Spaces can use it to
    # shorten demos). Default: the suite's full horizon.
    max_steps = int(os.environ.get("WEBVLA_SPACE_MAX_STEPS", env.max_steps))
    meta["max_steps"] = max_steps

    try:
        while env.steps < max_steps and not done:
            pobs = env.obs_pi05()
            t0 = time.perf_counter()
            chunk = infer_chunk(pobs["observation/image"],
                                pobs["observation/wrist_image"],
                                pobs["observation/state"],
                                instruction)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            for a in chunk:
                _, _, done, _ = env.step(a)
                frames.append(env.render("agentview"))
                if done:
                    break
            progress(min(env.steps / max_steps, 1.0),
                     desc=f"step {env.steps}/{max_steps}")
            yield (None, frames[-1],
                   f"running — step {env.steps}/{max_steps}, "
                   f"chunk {len(latencies)}, last inference "
                   f"{latencies[-1]:.0f} ms", "")
    except Exception as e:
        tb = traceback.format_exc()
        msg = str(e)
        meta.update(success=False, steps=env.steps, error=tb,
                    wall_time_s=round(time.time() - t_start, 2))
        mp4 = _save_episode(task_id, seed, frames, meta)
        if "quota" in msg.lower() or "zerogpu" in msg.lower():
            gr.Warning(f"ZeroGPU quota exhausted — {msg[:200]}")
            note = f"### OUT OF GPU QUOTA\n{msg[:300]}"
        else:
            note = f"### EPISODE CRASHED\n```\n{tb[-800:]}\n```"
        yield (mp4, frames[-1], note, _badge(meta))
        return

    success = env.check_success()
    meta.update(success=success, steps=env.steps,
                wall_time_s=round(time.time() - t_start, 2),
                n_replans=len(latencies),
                latency_mean_ms=round(float(np.mean(latencies)), 1),
                replan_latencies_ms=[round(x, 1) for x in latencies])
    mp4 = _save_episode(task_id, seed, frames, meta)
    yield (mp4, frames[-1], "done", _badge(meta))


def _badge(meta):
    if meta.get("instruction_overridden"):
        lines = ["### CUSTOM INSTRUCTION — success not verifiable",
                 f"- instruction (custom): *{meta['instruction']}*",
                 f"- scene's original task: *{meta['base_task_language']}*"]
    else:
        badge = "SUCCESS" if meta.get("success") else "FAILED (task not achieved)"
        lines = [f"### {badge}", f"- instruction: *{meta['instruction']}*"]
    lines.append(f"- steps: {meta.get('steps')} | wall: "
                 f"{meta.get('wall_time_s')}s | chunk-inference mean: "
                 f"{meta.get('latency_mean_ms', '?')} ms")
    return "\n".join(lines)


# ---------------------------------------------------------------- gallery
def _gallery_rows():
    rows = []
    for fn in sorted(os.listdir(EPISODES_DIR), reverse=True):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(EPISODES_DIR, fn)) as f:
                m = json.load(f)
        except Exception:
            continue
        date = (m.get("timestamp_utc") or "")[:16].replace("T", " ")
        succ = m.get("success")
        rows.append([m.get("task_id"), m.get("seed"),
                     "custom" if m.get("instruction_overridden")
                     else ("yes" if succ else "no"),
                     m.get("steps"), m.get("wall_time_s"),
                     m.get("latency_mean_ms"), date, m.get("mp4", "")])
    return rows


def _gallery_select(evt: gr.SelectData, rows):
    try:
        mp4 = rows[evt.index[0]][-1]
    except Exception:
        return None
    p = os.path.join(EPISODES_DIR, mp4)
    return p if mp4 and os.path.exists(p) else None


# ---------------------------------------------------------------- UI
_CSS = "#webvla-live img {image-rendering: pixelated;}"

with gr.Blocks(title="WebVLA — π0.5 LIBERO demo") as demo:
    gr.Markdown(
        "# WebVLA — π0.5 on LIBERO-Spatial (ZeroGPU)\n"
        "Pick a task, hit **Run episode**, and watch a π0.5 policy "
        "(LeRobot PyTorch port) drive a simulated robot arm in closed "
        "loop. Inference runs on ZeroGPU per action chunk; the physics sim "
        "runs on CPU. One episode at a time — if someone else is running, "
        "you queue.\n\n"
        "**GPU quota:** anonymous ~2 min/day, free login ~5 min/day, "
        "PRO ~40 min/day. An episode needs roughly 1-2 min of GPU. "
        "Episodes are recorded to the container's ephemeral disk — the "
        "gallery below resets when the Space restarts."
    )
    with gr.Row():
        task_dd = gr.Dropdown(choices=TASKS, label="LIBERO task (scene)",
                              scale=3)
        seed_nb = gr.Number(value=0, precision=0,
                            label="seed (init layout = seed % 50)", scale=1)
        run_btn = gr.Button("Run episode", variant="primary", scale=1)
    scene_img = gr.Image(label=None, height=180, interactive=False,
                         show_label=False)
    custom_tb = gr.Textbox(
        label="Custom instruction (optional) — same scene, your words; "
              "custom runs are UNCHECKED (success checker belongs to the "
              "original task)")
    status_md = gr.Markdown("idle — submit a job")
    with gr.Row():
        live_img = gr.Image(label="latest frame", elem_id="webvla-live",
                            interactive=False)
        out_video = gr.Video(label="episode")
    result_md = gr.Markdown("")
    gr.Markdown("### Past episodes (this container only)")
    gallery_df = gr.Dataframe(
        headers=["task", "seed", "success", "steps", "wall s",
                 "lat ms", "date (UTC)", "file"],
        value=_gallery_rows(), interactive=False, wrap=True)
    gallery_video = gr.Video(label="selected episode")

    def _scene(task_id):
        if task_id is None:
            return None
        p = os.path.join(SCENES_DIR, f"task{int(task_id)}.png")
        return p if os.path.exists(p) else None

    task_dd.change(_scene, task_dd, scene_img)
    run_btn.click(run_episode, [task_dd, seed_nb, custom_tb],
                  [out_video, live_img, status_md, result_md])
    gallery_df.select(_gallery_select, gallery_df, gallery_video)
    refresh_btn = gr.Button("Refresh gallery")
    refresh_btn.click(lambda: _gallery_rows(), None, gallery_df)


if __name__ == "__main__":
    # Warmup: first GPU call pays the weight-streaming cold start; do it at
    # boot so the first visitor doesn't. Best-effort — never block startup.
    try:
        _o = env.reset(0, 0)
        _p = env.obs_pi05()
        infer_chunk(_p["observation/image"], _p["observation/wrist_image"],
                    _p["observation/state"], _o["instruction"])
        print("[space] warmup inference OK", flush=True)
    except Exception as e:
        print(f"[space] warmup failed (continuing): {e}", flush=True)

    demo.queue(default_concurrency_limit=1).launch()
