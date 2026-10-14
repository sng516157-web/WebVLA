#!/usr/bin/env bash
/root/webvla/venv_lerobot/bin/python - <<'PYEOF'
import asyncio
import json
import uuid

import websockets


async def main():
    async with websockets.connect("ws://127.0.0.1:7870/ws") as ws:
        env = {
            "version": 1, "id": uuid.uuid4().hex, "type": "session.start",
            "sender": "client", "ts": 0,
            "payload": {"task_id": 0, "seed": 0, "instruction": None},
        }
        await ws.send(json.dumps(env))
        # collect messages for 12s
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=12)
                if isinstance(msg, bytes):
                    print("BINARY frame", len(msg), "bytes", flush=True)
                else:
                    print("TEXT:", msg[:300], flush=True)
        except (asyncio.TimeoutError, TimeoutError):
            print("timeout waiting", flush=True)


asyncio.run(main())
PYEOF
