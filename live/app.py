"""Live mode: interactive VLA episode streamed over WebSocket.

Standalone app (separate process from the queued M1/M2 server):
    bash setup/run.sh -m uvicorn live.app:app --host 0.0.0.0 --port 7870

One session, one viewer, localhost. No queue, no persistence, no recording.
The validated M0/M1/M2 pipeline is imported but never modified.
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

import paths
from live.session import LiveSession

INDEX_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

DEFAULT_CHECKPOINT = os.path.join(
    paths.CKPTS_DIR, "octo_finetune", "libero_spatial",
    "libero_spatial_bs16_lr5e5_20261010_114251",
)
SUITE_NAME = os.environ.get("WEBVLA_SUITE", "libero_spatial")


@asynccontextmanager
async def lifespan(app):
    session = LiveSession(
        checkpoint=os.environ.get("WEBVLA_CKPT", DEFAULT_CHECKPOINT),
        suite_name=SUITE_NAME,
    )
    session.start()
    await asyncio.to_thread(session.load)  # env class + checkpoint off the event loop
    app.state.session = session
    print("[live] session loaded; open http://localhost:7870/", flush=True)
    yield
    session.shutdown()


app = FastAPI(title="WebVLA live mode", lifespan=lifespan)


class StartRequest(BaseModel):
    task_id: int
    seed: int = 0
    instruction: str | None = None


class ResetRequest(BaseModel):
    task_id: int | None = None
    seed: int | None = None
    instruction: str | None = None


class ClickRequest(BaseModel):
    x: float
    y: float
    push: bool = True


@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


@app.get("/tasks")
async def tasks():
    session = app.state.session
    if session.wrapper is None:
        return {"ready": False, "tasks": []}
    return {
        "ready": True,
        "suite": session.suite_name,
        "tasks": [
            {"task_id": i, "language": t.language}
            for i, t in enumerate(session.wrapper.benchmark.tasks)
        ],
    }


@app.get("/session")
async def session_status():
    return app.state.session.status()


@app.post("/session/start")
async def session_start(req: StartRequest):
    session = app.state.session
    instruction = (req.instruction or "").strip()[:200] or None
    try:
        session.request_start(req.task_id, req.seed, instruction)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True}


@app.post("/session/reset")
async def session_reset(req: ResetRequest):
    session = app.state.session
    instruction = (req.instruction or "").strip()[:200] or None
    try:
        session.request_reset(req.task_id, req.seed, instruction)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True}


@app.post("/session/stop")
async def session_stop():
    app.state.session.request_stop()
    return {"ok": True}


@app.post("/click")
def click(req: ClickRequest):
    """Sync endpoint (runs in the threadpool): unproject + optional impulse."""
    session = app.state.session
    try:
        return session.handle_click(req.x, req.y, req.push)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    session = app.state.session
    hub = session.hub
    cid = hub.register()
    last_id = 0
    try:
        while True:
            # newest-frame-only: wait for a frame newer than the last one sent;
            # anything skipped in between counts as dropped for this client
            fid, jpeg, ts = await asyncio.to_thread(hub.wait_newer, last_id)
            if fid == last_id:
                continue
            import time
            age = time.time() - ts
            await websocket.send_bytes(jpeg)
            stats = hub.clients[cid]
            if last_id:
                stats["dropped"] += max(0, fid - last_id - 1)
            stats["sent"] += 1
            stats["age_sum"] += age
            stats["age_max"] = max(stats["age_max"], age)
            last_id = fid
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.unregister(cid)
