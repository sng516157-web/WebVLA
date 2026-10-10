"""Single persistent worker: env + policy loaded once, jobs run strictly
sequentially in one executor thread (blocking JAX/muJoCo never touches the
event loop).

Episode execution reuses M0 verbatim: runner.batch.build_stack +
runner.run_episode.run_and_save. run_and_save names outputs
`{suite}__task{T}__seed{S}` with no job id, so episodes run into a per-job
temp dir and are moved to `episodes/` with a `__{job_id}` suffix — M0 code
stays untouched and same task+seed jobs can't overwrite each other.

Failed jobs: the exception traceback is recorded on the job AND written as a
failure metadata JSON into episodes/ so the gallery never silently drops it.
"""

import asyncio
import json
import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from runner.batch import build_stack
from runner.run_episode import run_and_save
from server.jobs import STATUS_DONE, STATUS_FAILED, STATUS_RUNNING, JobStore
from sim.wrapper import LiberoWrapper


class Worker:
    def __init__(self, store: JobStore, checkpoint: str, episodes_dir: str,
                 suite: str = "libero_spatial", exec_horizon: int = 4,
                 backend: str = "gpu"):
        self.store = store
        self.checkpoint = checkpoint
        self.episodes_dir = episodes_dir
        self.suite = suite
        self.exec_horizon = exec_horizon
        self.backend = backend
        self.env = None
        self.policy = None
        self.warmup_done = False

    def build(self):
        """Blocking: construct env + policy, then JIT-warm one replan."""
        t0 = time.time()
        self.env, self.policy = build_stack(self.suite, self.checkpoint)
        obs = self.env.reset(0, 0)
        self.policy.reset(obs["instruction"])
        self.policy.act(self.env.obs())  # JIT compile; first call is slow
        self.warmup_done = True
        print(f"[worker] stack built + warmed in {time.time()-t0:.1f}s "
              f"(backend={self.backend}, checkpoint={self.checkpoint})")

    def _run_episode(self, job):
        """Blocking: run one episode via M0's run_and_save, return result."""
        tmp_dir = os.path.join(self.episodes_dir, f".tmp_{job.job_id}")
        os.makedirs(tmp_dir, exist_ok=True)
        extra_meta = {
            "job_id": job.job_id,
            "backend": self.backend,
            "checkpoint": self.checkpoint,
        }
        try:
            result = run_and_save(
                self.env, self.policy, self.suite, job.task_id, job.seed,
                tmp_dir, exec_horizon=self.exec_horizon, extra_meta=extra_meta,
            )
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        # Move outputs into episodes/ with a job-id suffix.
        base = f"{self.suite}__task{job.task_id}__seed{job.seed}__{job.job_id}"
        shutil.move(os.path.join(tmp_dir, result["mp4"]),
                    os.path.join(self.episodes_dir, base + ".mp4"))
        result["mp4"] = base + ".mp4"
        with open(os.path.join(self.episodes_dir, base + ".json"), "w") as f:
            json.dump(result, f, indent=2)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return result

    def _write_failure(self, job, tb):
        base = f"{self.suite}__task{job.task_id}__seed{job.seed}__{job.job_id}"
        meta = {
            "suite": self.suite,
            "task_id": job.task_id,
            "seed": job.seed,
            "job_id": job.job_id,
            "backend": self.backend,
            "success": False,
            "error": tb,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(self.episodes_dir, base + ".json"), "w") as f:
            json.dump(meta, f, indent=2)

    async def run(self):
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as ex:
            await loop.run_in_executor(ex, self.build)
            while True:
                job_id = await self.store.queue.get()
                job = await self.store.get(job_id)
                if job is None:
                    continue
                await self.store.set_status(job, STATUS_RUNNING)
                try:
                    job.result = await loop.run_in_executor(ex, self._run_episode, job)
                    await self.store.set_status(job, STATUS_DONE)
                    print(f"[worker] job {job.job_id} done: task={job.task_id} "
                          f"seed={job.seed} success={job.result['success']}")
                except Exception:
                    tb = traceback.format_exc()
                    job.error = tb
                    try:
                        self._write_failure(job, tb)
                    except Exception:
                        print(f"[worker] failed to write failure json for {job.job_id}")
                    await self.store.set_status(job, STATUS_FAILED)
                    print(f"[worker] job {job.job_id} FAILED (continuing):\n{tb}")
                    # A failed reset can leave the wrapper holding a closed or
                    # half-built env (get_task raises AFTER env.close()).
                    # Replace the wrapper wholesale — cheap, no model reload.
                    try:
                        self.env.close()
                    except Exception:
                        pass
                    self.env = LiberoWrapper(self.suite)
