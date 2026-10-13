"""FastAPI layer: POST /episode, GET /episode/{job_id}, GET /episodes.

One persistent worker (env + policy stay loaded between jobs) consumes an
asyncio.Queue; job state lives in an in-memory JobStore. Gradio is mounted at
"/" by web.ui (skipped when gradio isn't installed or WEBVLA_NO_UI=1).

Backend selection: WEBVLA_BACKEND=gpu|cpu (default: cpu on HF Spaces, gpu
elsewhere). cpu forces JAX_PLATFORMS=cpu before the stack is built; rendering
backend comes from MUJOCO_GL (egl locally, osmesa on Spaces).
"""

import asyncio
import glob
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import paths
from server.jobs import STATUS_QUEUED, STATUS_RUNNING, JobStore
from server.worker import Worker

EPISODES_DIR = os.path.join(paths.REPO_ROOT, "episodes")
LIVE_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "static", "live.html")

EPISODES_DIR = os.path.join(paths.REPO_ROOT, "episodes")
DEFAULT_CHECKPOINT = os.path.join(
    paths.CKPTS_DIR, "octo_finetune", "libero_spatial",
    "libero_spatial_bs16_lr5e5_20261010_114251",
)


def _backend() -> str:
    b = os.environ.get("WEBVLA_BACKEND")
    if b:
        return b
    return "cpu" if os.environ.get("SPACE_ID") else "gpu"


@asynccontextmanager
async def lifespan(app):
    os.makedirs(EPISODES_DIR, exist_ok=True)
    backend = _backend()
    if backend == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    store = JobStore()
    worker = Worker(
        store,
        checkpoint=os.environ.get("WEBVLA_CKPT", DEFAULT_CHECKPOINT),
        episodes_dir=EPISODES_DIR,
        backend=backend,
        policy_backend=os.environ.get("WEBVLA_POLICY", "octo"),
    )
    app.state.store = store
    app.state.worker = worker
    task = asyncio.create_task(worker.run())
    app.state.worker_task = task
    yield
    task.cancel()


app = FastAPI(title="WebVLA M1", lifespan=lifespan)


class SubmitRequest(BaseModel):
    task_id: int
    seed: int = 0
    instruction: str | None = None  # custom instruction on the task's scene


class InjectRequest(BaseModel):
    text: str


@app.post("/episode")
async def submit_episode(req: SubmitRequest):
    instruction = (req.instruction or "").strip()[:200] or None
    job = await app.state.store.submit(req.task_id, req.seed, instruction)
    app.state.worker.streams.for_job(job.job_id)  # ws clients can connect at once
    return {"job_id": job.job_id}


@app.get("/episode/{job_id}")
async def episode_status(job_id: str):
    store = app.state.store
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    resp = {
        "job_id": job.job_id,
        "task_id": job.task_id,
        "seed": job.seed,
        "instruction": job.instruction,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }
    if job.status == STATUS_QUEUED:
        resp["queue_position"] = await store.queue_position(job)
    if job.status == "done" and job.result is not None:
        resp["result"] = {
            "mp4_url": f"/episodes/{job.result['mp4']}",
            "metadata": job.result,
        }
    if job.status == "failed":
        resp["error"] = job.error
    if job.result and job.result.get("instruction_events"):
        resp["instruction_events"] = job.result["instruction_events"]
    return resp


@app.post("/episode/{job_id}/instruction")
async def inject_instruction(job_id: str, req: InjectRequest):
    """M2: queue an instruction swap for a RUNNING job (applied at the next
    replan boundary). 409 unless the job is currently running."""
    store = app.state.store
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    if job.status != STATUS_RUNNING:
        raise HTTPException(status_code=409, detail=f"job is {job.status}, not running")
    text = req.text.strip()[:200]
    if not text:
        raise HTTPException(status_code=400, detail="empty instruction")
    job.pending_instruction = text
    return {"ok": True, "queued_instruction": text}


@app.get("/episode/{job_id}/frame.jpg")
async def latest_frame(job_id: str):
    """Latest streamed frame for a job (polling fallback for the ws)."""
    bc = app.state.worker.streams.get(job_id)
    if bc is None or bc.latest_frame is None:
        return Response(status_code=204)
    return Response(content=bc.latest_frame, media_type="image/jpeg")


@app.websocket("/ws/jobs/{job_id}")
async def job_stream(ws: WebSocket, job_id: str):
    """M2 live stream: binary JPEG frames, then a terminal JSON text message."""
    bc = app.state.worker.streams.get(job_id)
    if bc is None:
        await ws.close(code=4404)
        return
    await ws.accept()
    q = bc.subscribe()
    try:
        while True:
            msg = await q.get()
            if isinstance(msg, bytes):
                await ws.send_bytes(msg)
            else:
                await ws.send_text(msg)
                break  # terminal control message ends the stream
    except WebSocketDisconnect:
        pass
    finally:
        bc.unsubscribe(q)
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/live")
async def live_page():
    """Live-view iframe page (server/static/live.html; job_id via query)."""
    return FileResponse(LIVE_HTML)


@app.get("/episodes")
async def list_episodes():
    """Gallery: every metadata JSON in episodes/, newest first."""
    out = []
    for path in glob.glob(os.path.join(EPISODES_DIR, "*.json")):
        try:
            with open(path) as f:
                meta = json.load(f)
        except Exception:
            continue
        mp4 = meta.get("mp4")
        out.append({
            "file": os.path.basename(path),
            "mp4_url": f"/episodes/{mp4}" if mp4 and os.path.exists(
                os.path.join(EPISODES_DIR, mp4)) else None,
            "task_id": meta.get("task_id"),
            "seed": meta.get("seed"),
            "success": meta.get("success"),
            "instruction": meta.get("instruction"),
            "instruction_overridden": meta.get("instruction_overridden", False),
            "base_task_language": meta.get("base_task_language"),
            "steps": meta.get("steps"),
            "wall_time_s": meta.get("wall_time_s"),
            "latency_mean_ms": meta.get("latency_mean_ms"),
            "timestamp_utc": meta.get("timestamp_utc"),
            "job_id": meta.get("job_id"),
            "error": meta.get("error"),
            "mtime": os.path.getmtime(path),
        })
    out.sort(key=lambda e: e["mtime"], reverse=True)
    for e in out:
        e.pop("mtime")
    return {"episodes": out}


@app.get("/tasks")
async def list_tasks():
    """Task list for the UI dropdown, from the worker's loaded benchmark."""
    worker = app.state.worker
    if worker.env is None:
        return {"ready": False, "tasks": []}
    return {
        "ready": True,
        "tasks": [
            {"task_id": i, "language": t.language}
            for i, t in enumerate(worker.env.benchmark.tasks)
        ],
    }


@app.get("/healthz")
async def healthz():
    worker = app.state.worker
    return {
        "ok": True,
        "backend": worker.backend,
        "warmup_done": worker.warmup_done,
        "worker_alive": not app.state.worker_task.done(),
    }


os.makedirs(EPISODES_DIR, exist_ok=True)  # needed at import time on fresh containers
app.mount("/episodes", StaticFiles(directory=EPISODES_DIR), name="episodes")

if os.environ.get("WEBVLA_NO_UI") != "1":
    try:
        import gradio as gr
        from web.ui import build_gradio

        app = gr.mount_gradio_app(
            app, build_gradio(), path="/",
            allowed_paths=[EPISODES_DIR, os.path.join(paths.REPO_ROOT, "assets")],
        )
    except ImportError:
        print("[server] gradio not installed — API-only mode")
