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
XRAY_JS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xray.js")

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
        policy_backend=os.environ.get("WEBVLA_POLICY", "octo"),
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


@app.get("/xray.js")
async def xray_js():
    return FileResponse(XRAY_JS, media_type="application/javascript")


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
    """One socket, three concerns:
      - binary JPEG frames out (each preceded by a "frame.meta" envelope),
        newest-frame-only, slow clients drop;
      - X-Ray event envelopes out (JSON text) via a per-client queue;
      - command envelopes in (JSON text) with state-intent dedupe and
        latest-wins continuous semantics (live/protocol.py).
    """
    import time as _time

    from live import protocol

    await websocket.accept()
    session = app.state.session
    hub = session.hub
    cid = hub.register()
    loop = asyncio.get_running_loop()
    events = asyncio.Queue()
    hub.register_events(cid, loop, events)

    def emit_error(msg):
        hub.broadcast_event(protocol.encode(
            "error", {"detail": msg[:300]}, "server"))

    router = protocol.CommandRouter()

    def _safe(fn):
        def wrapped(payload, msg):
            try:
                fn(payload, msg)
            except (RuntimeError, ValueError, KeyError) as e:
                emit_error(f"{msg['type']}: {e}")
        return wrapped

    router.register("xray.toggle", _safe(
        lambda p, m: session.xray.toggle(p["overlay"], bool(p["enabled"]))))
    router.register("session.start", _safe(
        lambda p, m: session.request_start(int(p["task_id"]),
                                           int(p.get("seed") or 0),
                                           p.get("instruction"))))
    router.register("session.reset", _safe(
        lambda p, m: session.request_reset(
            p.get("task_id"), p.get("seed"), p.get("instruction"))))
    router.register("session.stop", _safe(lambda p, m: session.request_stop()))

    def _click(p, m):
        res = session.handle_click(float(p["x"]), float(p["y"]),
                                   bool(p.get("push", True)))
        res["in_reply_to"] = m["id"]
        hub.broadcast_event(protocol.encode("click.result", res, "server"))

    router.register("pointer.click", _safe(_click))

    async def frame_sender():
        last_id = 0
        while True:
            fid, jpeg, ts = await asyncio.to_thread(hub.wait_newer, last_id)
            if fid == last_id:
                continue
            age = _time.time() - ts
            await websocket.send_text(protocol.encode(
                "frame.meta", {"frame_id": fid, "ts": ts}, "server"))
            await websocket.send_bytes(jpeg)
            stats = hub.clients[cid]
            if last_id:
                stats["dropped"] += max(0, fid - last_id - 1)
            stats["sent"] += 1
            stats["age_sum"] += age
            stats["age_max"] = max(stats["age_max"], age)
            last_id = fid

    async def event_sender():
        while True:
            await websocket.send_text(await events.get())

    async def cmd_receiver():
        while True:
            raw = await websocket.receive_text()
            try:
                msg = protocol.parse(raw)
            except protocol.ProtocolError as e:
                emit_error(f"bad envelope: {e}")
                continue
            await asyncio.to_thread(router.handle, msg)
            await asyncio.to_thread(router.drain_continuous)

    tasks = [asyncio.create_task(f()) for f in
             (frame_sender, event_sender, cmd_receiver)]
    try:
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in pending:
            t.cancel()
    except (WebSocketDisconnect, RuntimeError):
        for t in tasks:
            t.cancel()
    finally:
        hub.unregister(cid)
