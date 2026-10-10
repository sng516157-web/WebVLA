"""Step 4 — batch runner: N LIBERO tasks x M seeds -> episodes/ + results.json.

Prints per-task success rates next to the step-1 reference numbers
(reference/reference_results.json, if present).

Run inside WSL:
  MUJOCO_GL=osmesa /root/webvla/venv/bin/python run_batch.py
"""

import argparse
import json
import os

from runner.batch import aggregate, build_stack, run_grid, sanity_check


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-ids", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--checkpoint", default="hf://rail-berkeley/octo-base-1.5")
    p.add_argument("--exec-horizon", type=int, default=4)
    p.add_argument("--window-size", type=int, default=None,
                   help="None = infer from checkpoint example_batch (1 for finetuned)")
    p.add_argument("--out-dir", default=".",
                   help="run_grid appends 'episodes/' itself; '.' -> ./episodes/")
    p.add_argument("--results", default="results.json")
    p.add_argument("--reference", default="reference/reference_results.json")
    args = p.parse_args()

    env, policy = build_stack(args.suite, args.checkpoint, args.window_size)
    sanity = sanity_check(env, policy, task_id=args.task_ids[0], seed=args.seeds[0])

    results = run_grid(env, policy, args.suite, args.task_ids, args.seeds,
                       args.out_dir, args.exec_horizon, tag="batch")
    summary = aggregate(results)
    summary["config"] = vars(args)
    summary["sanity_check"] = sanity

    with open(args.results, "w") as f:
        json.dump(summary, f, indent=2)

    reference = None
    if os.path.exists(args.reference):
        with open(args.reference) as f:
            reference = json.load(f)

    print("\n=== BATCH RESULTS ===")
    print(f"{'task':>6}  {'batch':>10}  {'reference':>10}  instruction")
    for tid, t in summary["per_task"].items():
        ref = ""
        if reference:
            rt = reference.get("per_task", {}).get(str(tid))
            if rt:
                ref = f"{rt['successes']}/{rt['episodes']}"
        print(f"{tid:>6}  {t['successes']}/{t['episodes']:<8}  {ref:>10}  {t['instruction'][:60]}")
    print(f"overall: {summary['n_successes']}/{summary['n_episodes']} = {summary['overall_success_rate']:.1%}")
    if reference:
        print(f"reference overall: {reference['n_successes']}/{reference['n_episodes']} "
              f"= {reference['overall_success_rate']:.1%}")
    print(f"latency ms: {summary['latency_ms']}")
    print(f"failed episodes: {summary['failed_episodes']}")
    print(f"written to {args.results}")


if __name__ == "__main__":
    main()
