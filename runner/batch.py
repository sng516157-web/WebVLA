"""Shared helpers for the CLI entry points (reference_eval.py, run_batch.py)."""

import json
import os
import platform
import subprocess
import time

import numpy as np

from runner.run_episode import run_and_save


def build_stack(suite_name, checkpoint=None, window_size=None, unnorm_key=None,
                policy_backend="octo"):
    from sim.wrapper import LiberoWrapper

    env = LiberoWrapper(suite_name)
    if policy_backend == "octo":
        from policy.octo_policy import OctoPolicy

        policy = OctoPolicy(checkpoint, window_size=window_size, unnorm_key=unnorm_key)
    elif policy_backend == "pi05":
        from policy.pi0_policy import Pi05Policy

        policy = Pi05Policy()
    else:
        raise ValueError(f"unknown policy backend: {policy_backend!r}")
    return env, policy


def sanity_check(env, policy, task_id=0, seed=0):
    """Compare wrapper obs keys/shapes against what the policy consumes.

    Prints and returns a dict; mismatches would mean garbage actions, so this
    is part of the recorded results, not just stdout. Uses the policy's
    OBS_METHOD (obs for Octo, obs_pi05 for pi0.5) and its expected_observation.
    """
    obs_fn = getattr(env, getattr(policy, "OBS_METHOD", "obs"))
    obs = env.reset(task_id, seed)
    policy.reset(obs["instruction"])
    actual_obs = obs_fn()

    expected = policy.expected_observation  # e.g. image_primary: (1, window, 256, 256, 3)
    report = {"expected": {}, "actual": {}, "match": True}
    print("=" * 60)
    print("SANITY CHECK: obs camera keys vs policy-expected keys")
    for key, shape in expected.items():
        if not (key.startswith("image_") or key.startswith("observation/")):
            continue
        exp = tuple(shape)
        act = tuple(actual_obs[key].shape) if key in actual_obs else None
        # Octo expected shapes carry leading (batch, window) dims; compare HWC tail.
        exp_hwc = exp[-3:] if len(exp) > 3 or (len(exp) == 3 and key.startswith("image_")) else exp
        ok = act == exp_hwc or act == exp
        report["expected"][key] = list(exp_hwc)
        report["actual"][key] = list(act) if act else None
        report["match"] = report["match"] and ok
        print(f"  {key}: expected={exp_hwc}  actual={act}  {'OK' if ok else 'MISMATCH!'}")
    print(f"  RESULT: {'MATCH' if report['match'] else 'MISMATCH'}")
    print("=" * 60)
    return report


def gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def run_grid(env, policy, suite_name, task_ids, seeds, out_dir, exec_horizon, tag):
    episodes_dir = os.path.join(out_dir, "episodes")
    os.makedirs(episodes_dir, exist_ok=True)
    backend = "pi05" if getattr(policy, "OBS_METHOD", "obs") == "obs_pi05" else "octo"
    extra_meta = {
        "policy_backend": backend,
        "checkpoint": getattr(policy, "checkpoint", getattr(policy, "server", None)),
        "unnorm_key": getattr(policy, "unnorm_key", None),
        "window_size": getattr(policy, "window_size", None),
        "mujoco_gl": os.environ.get("MUJOCO_GL", "unset"),
        "gpu": gpu_name(),
        "host": platform.node(),
        "tag": tag,
    }
    results = []
    for task_id in task_ids:
        for seed in seeds:
            t0 = time.time()
            r = run_and_save(
                env, policy, suite_name, task_id, seed, episodes_dir,
                exec_horizon=exec_horizon, extra_meta=extra_meta,
            )
            results.append(r)
            print(
                f"[{tag}] suite={suite_name} task={task_id} seed={seed} "
                f"success={r['success']} steps={r['steps']} "
                f"replans={r['n_replans']} wall={r['wall_time_s']}s "
                f"(+{time.time()-t0:.1f}s total)"
            )
    return results


def aggregate(results):
    per_task = {}
    for r in results:
        t = per_task.setdefault(r["task_id"], {"successes": 0, "episodes": 0, "instruction": r["instruction"]})
        t["episodes"] += 1
        t["successes"] += int(r["success"])
    per_task_rate = {
        str(k): {"success_rate": v["successes"] / v["episodes"], "successes": v["successes"],
                 "episodes": v["episodes"], "instruction": v["instruction"]}
        for k, v in sorted(per_task.items())
    }
    all_lat = [l for r in results for l in r["replan_latencies_ms"]]
    failed = [f"task{r['task_id']}/seed{r['seed']}" for r in results if not r["success"]]
    total_succ = sum(int(r["success"]) for r in results)
    return {
        "n_episodes": len(results),
        "n_successes": total_succ,
        "overall_success_rate": total_succ / len(results) if results else None,
        "per_task": per_task_rate,
        "failed_episodes": failed,
        "latency_ms": {
            "mean": round(float(np.mean(all_lat)), 2) if all_lat else None,
            "p50": round(float(np.percentile(all_lat, 50)), 2) if all_lat else None,
            "p95": round(float(np.percentile(all_lat, 95)), 2) if all_lat else None,
            "max": round(float(np.max(all_lat)), 2) if all_lat else None,
        },
        "episodes": results,
    }
