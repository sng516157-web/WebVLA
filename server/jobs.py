"""In-memory job store: one job at a time, visible queue.

Jobs are plain dataclasses in a dict guarded by an asyncio.Lock; an
asyncio.Queue feeds the single persistent worker. Jobs survive individual
request/job failures — the worker records the traceback and moves on.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


@dataclass
class Job:
    job_id: str
    task_id: int
    seed: int
    instruction: str | None = None
    status: str = STATUS_QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self.queue: asyncio.Queue[str] = asyncio.Queue()

    async def submit(self, task_id: int, seed: int, instruction: str | None = None) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], task_id=task_id, seed=seed,
                  instruction=instruction)
        async with self._lock:
            self._jobs[job.job_id] = job
        await self.queue.put(job.job_id)
        return job

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def set_status(self, job: Job, status: str):
        async with self._lock:
            job.status = status
            if status == STATUS_RUNNING:
                job.started_at = time.time()
            if status in (STATUS_DONE, STATUS_FAILED):
                job.finished_at = time.time()

    async def queue_position(self, job: Job) -> int | None:
        """Number of queued jobs ahead of this one (None unless queued)."""
        if job.status != STATUS_QUEUED:
            return None
        async with self._lock:
            return sum(
                1
                for j in self._jobs.values()
                if j.status == STATUS_QUEUED and j.created_at < job.created_at
            )
