"""Gradio Blocks UI, mounted into the FastAPI app at "/".

The UI is a thin client of the HTTP API — Gradio's own queue is NOT used
(our JobStore is the only queue): Run -> POST /episode, a gr.Timer polls
GET /episode/{job_id} every 2s for queue position / progress / result, and
the gallery reads GET /episodes (every metadata JSON in episodes/, including
failures with their traceback).
"""

import os
from datetime import datetime

import gradio as gr
import httpx

import paths

API_BASE = os.environ.get(
    "WEBVLA_API_BASE", "http://127.0.0.1:" + os.environ.get("WEBVLA_PORT", "7860")
)
EPISODES_DIR = os.path.join(paths.REPO_ROOT, "episodes")
SCENES_DIR = os.path.join(paths.REPO_ROOT, "assets", "scenes")
M0_VALIDATED_TASKS = {0, 1, 2}


def _scene_img(task_id):
    if task_id is None:
        return None
    p = os.path.join(SCENES_DIR, f"task{int(task_id)}.png")
    return p if os.path.exists(p) else None

GALLERY_HEADERS = ["task", "seed", "success", "custom", "steps", "wall s", "lat ms", "date (UTC)", "job", "file"]


def _client():
    return httpx.Client(base_url=API_BASE, timeout=15)


def _task_choices():
    try:
        d = _client().get("/tasks").json()
    except Exception:
        return []
    return [
        (
            f"task {t['task_id']}"
            + (" (M0-validated)" if t["task_id"] in M0_VALIDATED_TASKS else "")
            + f" — {t['language']}",
            t["task_id"],
        )
        for t in d.get("tasks", [])
    ]


def _local_mp4(mp4_url):
    if not mp4_url:
        return None
    p = os.path.join(EPISODES_DIR, os.path.basename(mp4_url))
    return p if os.path.exists(p) else None


def _gallery_rows():
    try:
        eps = _client().get("/episodes").json()["episodes"]
    except Exception:
        return [], {}
    rows, by_row = [], {}
    for e in eps:
        date = (e.get("timestamp_utc") or "")[:16].replace("T", " ")
        succ = e.get("success")
        rows.append([
            e.get("task_id"), e.get("seed"),
            "?" if succ is None else ("yes" if succ else "no"),
            "yes" if e.get("instruction_overridden") else "",
            e.get("steps"), e.get("wall_time_s"), e.get("latency_mean_ms"),
            date, (e.get("job_id") or "")[:8], e.get("file"),
        ])
        by_row[e.get("file")] = e
    return rows, by_row


def _badge_md(m):
    """Result badge: custom-instruction episodes are UNCHECKED (the sim success
    checker still belongs to the scene's original task)."""
    if m.get("instruction_overridden"):
        return (
            "### CUSTOM INSTRUCTION — success not verifiable\n"
            f"- instruction (custom): *{m.get('instruction')}*\n"
            f"- scene's original task: *{m.get('base_task_language')}*"
        )
    badge = "SUCCESS" if m.get("success") else "FAILED (task not achieved)"
    return f"### {badge}\n- instruction: *{m.get('instruction')}*"


def submit(task_id, seed, instruction):
    if task_id is None:
        return None, "Pick a task first.", gr.update()
    try:
        job_id = _client().post(
            "/episode",
            json={"task_id": int(task_id), "seed": int(seed or 0),
                  "instruction": (instruction or "").strip() or None},
        ).json()["job_id"]
    except Exception as e:
        return None, f"submit failed: {e}", gr.update()
    return job_id, f"Submitted job `{job_id}` — waiting for status…", gr.update(value=None)


def poll(job_id, tick, dd_choices):
    """Timer handler: refresh job status; periodically gallery + task list."""
    status_md = gr.update()
    video = gr.update()
    stats = gr.update()
    error = gr.update()
    gallery = gr.update()
    dd = gr.update()
    new_job = job_id

    if not dd_choices:
        choices = _task_choices()
        if choices:
            dd = gr.update(choices=choices)

    if tick % 5 == 0:
        rows, _ = _gallery_rows()
        gallery = gr.update(value=rows)

    if job_id:
        try:
            d = _client().get(f"/episode/{job_id}").json()
        except Exception as e:
            return new_job, f"poll error: {e}", video, stats, error, gallery, dd
        st = d.get("status")
        if st == "queued":
            status_md = f"**queued** — position {d.get('queue_position')}"
        elif st == "running":
            status_md = "**running** — episode in progress (closed loop)…"
        elif st == "done":
            m = d["result"]["metadata"]
            status_md = f"**done** — job `{job_id}`"
            video = gr.update(value=_local_mp4(d["result"].get("mp4_url")))
            stats = _badge_md(m) + (
                f"\n- steps: {m.get('steps')} | replans: {m.get('n_replans')} "
                f"| wall: {m.get('wall_time_s')}s\n"
                f"- replan latency mean/p50/max: {m.get('latency_mean_ms')} / "
                f"{m.get('latency_p50_ms')} / {m.get('latency_max_ms')} ms"
            )
            new_job = None
            rows, _ = _gallery_rows()
            gallery = gr.update(value=rows)
        elif st == "failed":
            status_md = f"**failed** — job `{job_id}` crashed"
            error = gr.update(value=d.get("error") or "unknown error", visible=True)
            new_job = None
            rows, _ = _gallery_rows()
            gallery = gr.update(value=rows)
    return new_job, status_md, video, stats, error, gallery, dd


def on_gallery_select(evt: gr.SelectData):
    rows, by_row = _gallery_rows()
    try:
        fname = rows[evt.index[0]][-1]
    except Exception:
        return gr.update(), gr.update(), gr.update()
    e = by_row.get(fname) or {}
    if e.get("error"):
        return gr.update(value=None), gr.update(value=""), gr.update(
            value=e["error"], visible=True
        )
    stats = _badge_md(e) + (
        f"\n- steps: {e.get('steps')} | wall: {e.get('wall_time_s')}s "
        f"| replan latency mean: {e.get('latency_mean_ms')} ms"
    )
    return gr.update(value=_local_mp4(e.get("mp4_url"))), gr.update(value=stats), gr.update(visible=False)


def build_gradio():
    with gr.Blocks(title="WebVLA — LIBERO + Octo demo") as demo:
        gr.Markdown(
            "# WebVLA — queue-mode VLA demo\n"
            "Pick a LIBERO-Spatial task, submit it, and watch the recorded episode "
            "when your job finishes. One job runs at a time; the rest queue. "
            "Policy: Octo fine-tuned on LIBERO-Spatial (M0-validated closed loop, "
            "73.3% batch success rate)."
        )
        job_state = gr.State(None)
        tick_state = gr.State(0)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    task_dd = gr.Dropdown(
                        label="LIBERO task (scene)", choices=_task_choices(),
                        value=0, scale=4,
                    )
                    seed_in = gr.Number(label="seed (init layout = seed % 50)",
                                        value=0, precision=0, scale=1)
                    run_btn = gr.Button("Run episode", variant="primary", scale=1)
                custom_in = gr.Textbox(
                    label="Custom instruction (optional) — same scene, your words",
                    placeholder="e.g. pick up the ramekin and place it on the plate",
                    max_lines=1,
                )
                gr.Markdown(
                    "*Note: the policy was fine-tuned on only the 10 scene "
                    "instructions above, so custom phrasings often degrade to the "
                    "scene's default behavior — that generalization gap is part of "
                    "what this demo shows. Custom runs are UNCHECKED (the success "
                    "checker belongs to the original task). Reference the objects "
                    "you can see in the scene preview for the best shot.*"
                )
            scene_img = gr.Image(value=_scene_img(0), label="scene preview",
                                 height=280, width=280, interactive=False, scale=1)

        status_md = gr.Markdown("idle — submit a job")
        stats_md = gr.Markdown("")
        video = gr.Video(label="episode", height=400)
        error_tb = gr.Textbox(label="error traceback", visible=False, lines=12)

        gr.Markdown("### Past episodes (current container)")
        gallery = gr.Dataframe(
            headers=GALLERY_HEADERS, value=_gallery_rows()[0],
            interactive=False, wrap=True,
        )

        run_btn.click(submit, [task_dd, seed_in, custom_in], [job_state, status_md, video])
        task_dd.change(_scene_img, task_dd, scene_img)

        timer = gr.Timer(2)
        timer.tick(
            lambda j, t, c: (t + 1, *poll(j, t + 1, c)),
            [job_state, tick_state, task_dd],
            [tick_state, job_state, status_md, video, stats_md, error_tb, gallery, task_dd],
        )

        gallery.select(on_gallery_select, None, [video, stats_md, error_tb])
    return demo
