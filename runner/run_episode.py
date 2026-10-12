"""Closed-loop episode runner.

`run_episode_frames` is a Python generator that yields mp4 frames (HWC uint8)
while running the policy in closed loop: obs -> policy -> action chunk ->
execute `exec_horizon` steps -> replan. Episode statistics are written into the
`result` dict passed by the caller.

`run_and_save` consumes the generator, writes the mp4 (imageio/ffmpeg) and the
metadata JSON, and returns the completed result dict.
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np


def run_episode_frames(env, policy, task_id, seed, max_steps, exec_horizon, result,
                       hooks=None):
    """Generator: yields frames; fills `result` on completion.

    hooks (optional, M2): object with
      - on_frame(frame, step): called after every yielded frame
      - pending_instruction() -> str | None: polled at each replan boundary;
        a non-None value is adopted by the policy mid-episode and recorded in
        result["instruction_events"]. The sim success checker is untouched, so
        injected episodes are UNCHECKED by convention.
    hooks=None reproduces the M0 behavior exactly.
    """
    t_start = time.time()
    obs = env.reset(task_id, seed)
    policy.reset(obs["instruction"])
    latencies = []
    instruction_events = []
    done = False
    first_frame = env.render("agentview")
    if hooks is not None:
        hooks.on_frame(first_frame, env.steps)
    yield first_frame
    while env.steps < max_steps and not done:
        if hooks is not None:
            new_instr = hooks.pending_instruction()
            if new_instr:
                policy.set_instruction(new_instr)
                env.instruction = new_instr
                instruction_events.append({
                    "step": env.steps,
                    "text": new_instr,
                    "at_s": round(time.time() - t_start, 2),
                })
        t0 = time.perf_counter()
        actions = policy.act(env.obs())
        latencies.append((time.perf_counter() - t0) * 1000.0)
        for a in actions[:exec_horizon]:
            _, _, done, _ = env.step(a)
            frame = env.render("agentview")
            if hooks is not None:
                hooks.on_frame(frame, env.steps)
            yield frame
            if done:
                break
    lat_ms = [round(x, 2) for x in latencies]
    result.update(
        {
            "instruction": env.instruction,
            "seed": seed,
            "init_state_idx": env.init_state_idx,
            "success": env.check_success(),
            "steps": env.steps,
            "wall_time_s": round(time.time() - t_start, 2),
            "n_replans": len(latencies),
            "replan_latencies_ms": lat_ms,
            "latency_mean_ms": round(float(np.mean(lat_ms)), 2) if lat_ms else None,
            "latency_p50_ms": round(float(np.percentile(lat_ms, 50)), 2) if lat_ms else None,
            "latency_max_ms": round(float(np.max(lat_ms)), 2) if lat_ms else None,
        }
    )
    if instruction_events:
        result["instruction_events"] = instruction_events


def run_and_save(env, policy, suite_name, task_id, seed, out_dir, exec_horizon,
                 max_steps=None, fps=30, extra_meta=None, hooks=None):
    """Run one episode, save mp4 + metadata JSON into out_dir, return result."""
    max_steps = max_steps or env.max_steps
    name = f"{suite_name}__task{task_id}__seed{seed}"
    mp4_path = os.path.join(out_dir, name + ".mp4")
    json_path = os.path.join(out_dir, name + ".json")

    result = {
        "suite": suite_name,
        "task_id": task_id,
        "seed": seed,
        "max_steps": max_steps,
        "exec_horizon": exec_horizon,
        "fps": fps,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra_meta:
        result.update(extra_meta)

    import imageio.v2 as imageio

    writer = imageio.get_writer(mp4_path, fps=fps)
    frames = run_episode_frames(env, policy, task_id, seed, max_steps, exec_horizon,
                                result, hooks=hooks)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    result["mp4"] = os.path.basename(mp4_path)
    result["n_frames"] = env.steps + 1
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    return result
