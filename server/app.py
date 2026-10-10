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

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import paths
from server.jobs import STATUS_QUEUED, JobStore
from server.worker import Worker

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


@app.post("/episode")
async def submit_episode(req: SubmitRequest):
    job = await app.state.store.submit(req.task_id, req.seed)
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
    return resp


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


app.mount("/episodes", StaticFiles(directory=EPISODES_DIR), name="episodes")

if os.environ.get("WEBVLA_NO_UI") != "1":
    try:
        import gradio as gr
        from web.ui import build_gradio

        app = gr.mount_gradio_app(
            app, build_gradio(), path="/", allowed_paths=[EPISODES_DIR]
        )
    except ImportError:
        print("[server] gradio not installed — API-only mode")
