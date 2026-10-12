"""M0 regression: run_and_save WITHOUT hooks must behave exactly as before M2.

Runs one episode through the M0 code path (no hooks) and asserts the metadata
has the standard keys, no instruction_events, and frames/mp4 are produced.

  bash setup/run.sh setup/test_m0_regression.py
"""

import os
import tempfile

import paths
from runner.batch import build_stack
from runner.run_episode import run_and_save

CKPT = os.path.join(
    paths.CKPTS_DIR, "octo_finetune", "libero_spatial",
    "libero_spatial_bs16_lr5e5_20261010_114251",
)

EXPECTED_KEYS = {
    "suite", "task_id", "seed", "max_steps", "exec_horizon", "fps",
    "timestamp_utc", "instruction", "init_state_idx", "success", "steps",
    "wall_time_s", "n_replans", "replan_latencies_ms", "latency_mean_ms",
    "latency_p50_ms", "latency_max_ms", "mp4", "n_frames",
}


def main():
    env, policy = build_stack("libero_spatial", CKPT)
    with tempfile.TemporaryDirectory() as d:
        result = run_and_save(env, policy, "libero_spatial", 1, 0, d,
                              exec_horizon=4)
        missing = EXPECTED_KEYS - set(result)
        assert not missing, f"missing metadata keys: {missing}"
        assert "instruction_events" not in result
        assert os.path.exists(os.path.join(d, result["mp4"]))
        print(f"[OK] metadata keys intact; steps={result['steps']} "
              f"success={result['success']} mp4={result['mp4']}")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
