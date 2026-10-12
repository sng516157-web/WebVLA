"""Per-job live broadcaster for M2 streaming.

The worker runs episodes in an executor thread; WebSocket subscribers live on
the event loop. Broadcaster.publish() bridges the two with
loop.call_soon_threadsafe. Subscriber queues are bounded and drop the oldest
frame when full — a slow client can never back-pressure the rollout.

Messages: bytes = JPEG frame; str = JSON control message ({"type": ...}).
"""

import asyncio
import time

MAX_QUEUE = 30          # frames buffered per subscriber before drop-oldest
REMOVE_GRACE_S = 300    # keep a finished job's latest frame this long


class Broadcaster:
    def __init__(self, job_id):
        self.job_id = job_id
        self.latest_frame: bytes | None = None
        self.terminal: str | None = None  # JSON control message once ended
        self.subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self):
        self._loop = asyncio.get_running_loop()

    # --- producer side (any thread) ---

    def publish(self, msg):
        if isinstance(msg, bytes):
            self.latest_frame = msg
        else:
            self.terminal = msg if self.terminal is None else self.terminal
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._fanout, msg)

    def _fanout(self, msg):
        for q in list(self.subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    # --- consumer side (event loop) ---

    def subscribe(self) -> asyncio.Queue:
        self.bind_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE)
        if self.latest_frame is not None:
            q.put_nowait(self.latest_frame)
        if self.terminal is not None:
            q.put_nowait(self.terminal)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q):
        self.subscribers.discard(q)


class BroadcasterRegistry:
    def __init__(self):
        self._by_job: dict[str, Broadcaster] = {}
        self._remove_at: dict[str, float] = {}

    def for_job(self, job_id) -> Broadcaster:
        b = self._by_job.get(job_id)
        if b is None:
            b = Broadcaster(job_id)
            self._by_job[job_id] = b
        return b

    def get(self, job_id) -> Broadcaster | None:
        return self._by_job.get(job_id)

    def finish(self, job_id):
        """Mark for deferred removal (latest frame stays viewable briefly)."""
        self._remove_at[job_id] = time.time() + REMOVE_GRACE_S

    def sweep(self):
        now = time.time()
        for job_id, at in list(self._remove_at.items()):
            if now >= at and not self._by_job[job_id].subscribers:
                del self._by_job[job_id]
                del self._remove_at[job_id]
