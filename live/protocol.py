"""Envelope JSON protocol for live-mode X-Ray (see XRAY_MODEL.md / plan).

Every client<->server WebSocket message is an envelope:

    {"version": 1, "id": "<uuid4>", "type": "<dotted.type>",
     "sender": "client"|"server", "ts": <unix seconds>, "payload": {...}}

Two command categories, enforced server-side by CATEGORY:

  STATE_INTENT  — idempotent, deduplicated by envelope id (last wins):
                  "session.start", "session.reset", "session.stop",
                  "xray.toggle" (payload: {overlay: str, enabled: bool})
  CONTINUOUS    — latest-overrides-previous, loss-tolerant, never queued:
                  "pointer.click" (payload: {x, y, push}), reserved for
                  hover previews.

Server->client event types (all envelopes over the same socket):
  "frame.meta"  — precedes each binary JPEG frame: {frame_id, ts}
  "xray.chunk"  — per replan: {replan, latency_ms, exec_len, chunk_len,
                  waypoints_px: [[x,y],...]}
  "xray.exec"   — per executed step: {replan, k, action: [7]}
  "xray.denoise"— per replan: {replan, trajectories_px: [[[x,y]...] x10]}
  "xray.heatmap"— per replan: {replan, view, cols: 16, grid: [256]}
  "xray.bundle" — on failure introspection save: {replan, path, kept}
  "session.state"— status snapshot on change
"""

import json
import time
import uuid

VERSION = 1

STATE_INTENT = "state_intent"
CONTINUOUS = "continuous"

CATEGORY = {
    "session.start": STATE_INTENT,
    "session.reset": STATE_INTENT,
    "session.stop": STATE_INTENT,
    "xray.toggle": STATE_INTENT,
    "pointer.click": CONTINUOUS,
}


class ProtocolError(ValueError):
    pass


def envelope(msg_type, payload, sender):
    return {
        "version": VERSION,
        "id": uuid.uuid4().hex,
        "type": msg_type,
        "sender": sender,
        "ts": time.time(),
        "payload": payload or {},
    }


def encode(msg_type, payload, sender):
    return json.dumps(envelope(msg_type, payload, sender))


def parse(raw):
    try:
        msg = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"not JSON: {e}")
    if not isinstance(msg, dict):
        raise ProtocolError("envelope is not an object")
    if msg.get("version") != VERSION:
        raise ProtocolError(f"bad version: {msg.get('version')!r}")
    if not isinstance(msg.get("type"), str) or not msg["type"]:
        raise ProtocolError("missing type")
    if not isinstance(msg.get("id"), str) or not msg["id"]:
        raise ProtocolError("missing id")
    if not isinstance(msg.get("payload"), dict):
        raise ProtocolError("payload is not an object")
    return msg


class CommandRouter:
    """Server-side dispatch with the two command semantics.

    - state-intent: dedupe by envelope id (bounded), dispatch immediately.
    - continuous: buffer newest per type; `drain_continuous()` hands the
      newest snapshot to the handler and discards the rest — never queues.
    """

    _DEDUPE_MAX = 512

    def __init__(self):
        self._handlers = {}
        self._seen_ids = []
        self._seen_set = set()
        self._pending = {}  # type -> newest envelope

    def register(self, msg_type, handler):
        self._handlers[msg_type] = handler

    def handle(self, msg):
        t = msg["type"]
        cat = CATEGORY.get(t)
        if cat is None:
            raise ProtocolError(f"unknown message type {t!r}")
        if t not in self._handlers:
            return  # no handler bound: safely ignored
        if cat == STATE_INTENT:
            if msg["id"] in self._seen_set:
                return  # duplicate state-intent: harmless by design
            self._seen_set.add(msg["id"])
            self._seen_ids.append(msg["id"])
            if len(self._seen_ids) > self._DEDUPE_MAX:
                self._seen_set.discard(self._seen_ids.pop(0))
            self._handlers[t](msg["payload"], msg)
        else:  # CONTINUOUS: newest wins
            self._pending[t] = msg

    def drain_continuous(self):
        pending, self._pending = self._pending, {}
        for t, msg in pending.items():
            h = self._handlers.get(t)
            if h:
                h(msg["payload"], msg)
