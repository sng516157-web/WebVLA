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
import io
import json
import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from runner.batch import build_stack
from runner.run_episode import run_and_save
from server.custom_env import CustomInstructionWrapper
from server.jobs import STATUS_DONE, STATUS_FAILED, STATUS_RUNNING, JobStore
from server.stream import BroadcasterRegistry


class _Hooks:
    """M2 live hooks bound to one job: frame broadcast + instruction pickup."""

    def __init__(self, job, broadcaster):
        self.job = job
        self.bc = broadcaster

    def on_frame(self, frame, step):
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=80)
        self.bc.publish(buf.getvalue())

    def pending_instruction(self):
        text = self.job.pending_instruction
        self.job.pending_instruction = None
        return text


class Worker:
    def __init__(self, store: JobStore, checkpoint: str, episodes_dir: str,
                 suite: str = "libero_spatial", exec_horizon: int | None = None,
                 backend: str = "gpu", policy_backend: str = "octo"):
        self.store = store
        self.checkpoint = checkpoint
        self.episodes_dir = episodes_dir
        self.suite = suite
        self.policy_backend = policy_backend
        self.exec_horizon = exec_horizon if exec_horizon is not None else (
            5 if policy_backend == "pi05" else 4)
        self.backend = backend
        self.env = None
        self.policy = None
        self.warmup_done = False
        self.streams = BroadcasterRegistry()

    def build(self):
        """Blocking: construct env + policy, then JIT-warm one replan."""
        t0 = time.time()
        self.env, self.policy = build_stack(self.suite, self.checkpoint,
                                            policy_backend=self.policy_backend)
        # Same-layout subclass swap: keeps build_stack verbatim while enabling
        # per-job custom instructions (M0 code untouched).
        self.env.__class__ = CustomInstructionWrapper
        obs = self.env.reset(0, 0)
        self.policy.reset(obs["instruction"])
        obs_fn = getattr(self.env, getattr(self.policy, "OBS_METHOD", "obs"))
        self.policy.act(obs_fn())  # JIT compile; first call is slow
        self.warmup_done = True
        print(f"[worker] stack built + warmed in {time.time()-t0:.1f}s "
              f"(backend={self.backend}, policy={self.policy_backend}, "
              f"checkpoint={self.checkpoint})")

    def _run_episode(self, job):
        """Blocking: run one episode via M0's run_and_save, return result."""
        tmp_dir = os.path.join(self.episodes_dir, f".tmp_{job.job_id}")
        os.makedirs(tmp_dir, exist_ok=True)
        extra_meta = {
            "job_id": job.job_id,
            "backend": self.backend,
            "policy_backend": self.policy_backend,
            "checkpoint": self.checkpoint if self.policy_backend == "octo"
            else getattr(self.policy, "server", "pi05"),
        }
        self.env.custom_instruction = job.instruction
        extra_meta["base_task_language"] = self.env.benchmark.get_task(
            job.task_id).language
        if job.instruction:
            extra_meta["instruction_overridden"] = True
        bc = self.streams.for_job(job.job_id)
        try:
            result = run_and_save(
                self.env, self.policy, self.suite, job.task_id, job.seed,
                tmp_dir, exec_horizon=self.exec_horizon, extra_meta=extra_meta,
                hooks=_Hooks(job, bc),
            )
        except Exception:
            bc.publish(json.dumps({"type": "failed"}))
            self.streams.finish(job.job_id)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        finally:
            self.env.custom_instruction = None
        # Move outputs into episodes/ with a job-id suffix.
        base = f"{self.suite}__task{job.task_id}__seed{job.seed}__{job.job_id}"
        shutil.move(os.path.join(tmp_dir, result["mp4"]),
                    os.path.join(self.episodes_dir, base + ".mp4"))
        result["mp4"] = base + ".mp4"
        with open(os.path.join(self.episodes_dir, base + ".json"), "w") as f:
            json.dump(result, f, indent=2)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        bc.publish(json.dumps({"type": "done", "success": result.get("success"),
                               "steps": result.get("steps")}))
        self.streams.finish(job.job_id)
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
                    self.env = CustomInstructionWrapper(self.suite)
                self.streams.sweep()
