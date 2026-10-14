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


def _pip_no_deps(*pkgs):
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                    "--quiet", *pkgs], check=True)


def _ensure_lerobot():
    # Everything here is --no-deps, because the Space's pip resolution must
    # satisfy gradio 6.30's huggingface-hub>=1.16 while lerobot's pip deps
    # force numpy>=2 (breaks gym 0.25.2), the transformers fork declares
    # hub<1.0, and tokenizers 0.21.x declares hub<1.0. All real deps are in
    # requirements.txt; policy/_hub_compat.py relaxes the fork's runtime
    # version gate before the first transformers import.
    try:
        import lerobot  # noqa: F401
    except ImportError:
        print("[space] installing lerobot==0.4.4 (--no-deps) ...", flush=True)
        _pip_no_deps("lerobot==0.4.4")
    # tokenizers must be in place before the fork is imported (version gate
    # checks it); 0.21.4 matches the fork's declared >=0.21,<0.22.
    try:
        import tokenizers
        assert tokenizers.__version__ == "0.21.4"
    except (ImportError, AssertionError):
        print("[space] installing tokenizers==0.21.4 (--no-deps) ...",
              flush=True)
        _pip_no_deps("tokenizers==0.21.4")
    if not _fork_transformers_installed():
        print("[space] installing transformers fork (--no-deps) ...", flush=True)
        _pip_no_deps("--force-reinstall",
                     "git+https://github.com/huggingface/transformers.git"
                     "@fix/lerobot_openpi")


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
def infer_chunk(img, wrist, state, instruction, xray=False):
    """One action chunk on the GPU worker. Numpy in, numpy out.

    xray=True additionally returns the full 50x7 chunk and read-only
    internals (denoise trajectory + pooled 16x16 attention heatmaps, ~16 KB
    total). Observation-only: proven action-identical to the plain path by
    setup/xray_equivalence_check.py (max |dAction| = 0.0).
    """
    obs = {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
    }
    policy.reset(instruction)
    if not xray:
        return policy.act(obs)
    return policy.act_with_internals(obs)


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


def run_episode(task_id, seed, custom_instruction, xray_overlays=None,
                progress=gr.Progress()):
    """Gradio generator: streams progress, ends with the recorded video."""
    try:
        from xray_overlay import (  # Space staging: renamed to repo root
            chunk_waypoints_world, composite, downsample, make_projector)
    except ImportError:  # running from the repo checkout (space/ subdir)
        from space.xray_overlay import (
            chunk_waypoints_world, composite, downsample, make_projector)
    xray_overlays = xray_overlays or []
    xray = bool(xray_overlays)
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
    if xray:
        meta["xray_overlays"] = list(xray_overlays)

    frames = [env.render("agentview")]
    latencies = []
    ticks = []  # X-Ray replan timeline: inference ms per tick
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
            if xray:
                chunk, full_chunk, internals = infer_chunk(
                    pobs["observation/image"], pobs["observation/wrist_image"],
                    pobs["observation/state"], instruction, True)
            else:
                chunk = infer_chunk(pobs["observation/image"],
                                    pobs["observation/wrist_image"],
                                    pobs["observation/state"], instruction)
                full_chunk = internals = None
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)
            ticks.append(lat_ms)

            wps_px = den_px = heat = None
            if xray:
                project = make_projector(env)
                eef = np.asarray(env._raw["robot0_eef_pos"],
                                 dtype=np.float64)
                if "Action path" in xray_overlays:
                    wps_px = downsample(project(
                        chunk_waypoints_world(full_chunk, eef)))
                if ("Denoising steps" in xray_overlays
                        and internals.get("denoise_xt") is not None):
                    den_px = [downsample(project(chunk_waypoints_world(xt, eef)))
                              for xt in internals["denoise_xt"]]
                if "Grounding heatmap" in xray_overlays:
                    heat = internals.get("heatmap_agentview")

            for k, a in enumerate(chunk):
                _, _, done, _ = env.step(a)
                frame = env.render("agentview")
                if xray:
                    frame = composite(
                        frame,
                        waypoints=wps_px,
                        progress=((k + 1) / len(chunk))
                                 * (len(chunk) / len(full_chunk)),
                        action=a if "Action bars" in xray_overlays else None,
                        denoise=den_px,
                        heatmap=heat,
                        ticks=ticks if "Replan timeline" in xray_overlays
                        else None,
                        cur_replan=len(ticks) - 1)
                frames.append(frame)
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


def _gallery_select(evt: gr.SelectData):
    # Do NOT take gallery_df as an input here: Gradio preprocesses a
    # Dataframe input into a pandas DataFrame (type="pandas" default), so
    # rows[evt.index[0]] raises KeyError on the string headers — silently,
    # behind the try/except. The filesystem is the source of truth anyway.
    if evt is None or evt.index is None:
        return None
    rows = _gallery_rows()
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
    xray_cb = gr.CheckboxGroup(
        choices=["Action path", "Action bars", "Replan timeline",
                 "Denoising steps", "Grounding heatmap"],
        value=[], label="X-Ray overlays (opt-in) — drawn into the episode")
    gr.Markdown(
        "*What they are:* action path = the predicted next-50-action "
        "end-effector path (the policy executes 5, then replans); denoising "
        "= how the trajectory was refined over 10 flow-matching steps (not "
        "'intent'); heatmap = where instruction tokens attend (attention "
        "weights are *not* an explanation of why it acted). Overlays cost "
        "a little extra GPU time per replan.")
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
    run_btn.click(run_episode, [task_dd, seed_nb, custom_tb, xray_cb],
                  [out_video, live_img, status_md, result_md]
                  ).then(lambda: _gallery_rows(), None, gallery_df)
    gallery_df.select(_gallery_select, None, gallery_video)
    refresh_btn = gr.Button("Refresh gallery")
    refresh_btn.click(lambda: _gallery_rows(), None, gallery_df)
    # value=_gallery_rows() above is evaluated once at app startup; reload
    # per page load so visitors see episodes recorded after boot.
    demo.load(lambda: _gallery_rows(), None, gallery_df)


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
