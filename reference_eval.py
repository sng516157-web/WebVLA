"""Step 1 — reference validation.

Runs the Octo base checkpoint in closed loop on LIBERO (2 tasks x 3 episodes
by default) and records success rates to reference/reference_results.json.

IMPORTANT: octo-models/octo ships no LIBERO eval script (verified: main,
main_old branch, all tags, no file matching *libero* in repo history). This
script is the reference implementation: environment side follows the standard
LIBERO eval harness conventions (openvla/openvla experiments/robot/libero),
policy side follows octo's official examples/03_eval_finetuned.py verbatim.

Run inside WSL:
  MUJOCO_GL=osmesa /root/webvla/venv/bin/python reference_eval.py
"""

import argparse
import json
import os

from runner.batch import aggregate, build_stack, run_grid, sanity_check


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-ids", type=int, nargs="+", default=[0, 1])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--checkpoint", default="hf://rail-berkeley/octo-base-1.5")
    p.add_argument("--exec-horizon", type=int, default=4,
                   help="replan interval: steps of the predicted chunk executed open-loop")
    p.add_argument("--window-size", type=int, default=2)
    p.add_argument("--out-dir", default="reference")
    args = p.parse_args()

    env, policy = build_stack(args.suite, args.checkpoint, args.window_size)

    sanity = sanity_check(env, policy, task_id=args.task_ids[0], seed=args.seeds[0])

    results = run_grid(env, policy, args.suite, args.task_ids, args.seeds,
                       args.out_dir, args.exec_horizon, tag="reference")
    summary = aggregate(results)
    summary["config"] = vars(args)
    summary["sanity_check"] = sanity

    out_path = os.path.join(args.out_dir, "reference_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== REFERENCE RESULTS ===")
    for tid, t in summary["per_task"].items():
        print(f"  task {tid} ({t['instruction'][:60]}): {t['successes']}/{t['episodes']}")
    print(f"  overall: {summary['n_successes']}/{summary['n_episodes']} "
          f"= {summary['overall_success_rate']:.1%}")
    print(f"  latency ms: {summary['latency_ms']}")
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
