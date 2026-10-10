"""Step-0 gate: CPU inference latency benchmark.

Simulates the HF Space free tier (2 vCPU, no GPU): run under
  CUDA_VISIBLE_DEVICES="" taskset -c 0,1 bash setup/run.sh setup/bench_cpu_latency.py
Measures wall time of OctoModel load and per-replan policy.act latency.
"""

import time

from runner.batch import build_stack

t0 = time.perf_counter()
env, policy = build_stack("libero_spatial", "hf://rail-berkeley/octo-base-1.5")
print(f"model+env load: {time.perf_counter()-t0:.1f}s")

obs = env.reset(0, 0)
policy.reset(obs["instruction"])

lat = []
for i in range(4):
    t0 = time.perf_counter()
    policy.act(env.obs())
    lat.append((time.perf_counter() - t0) * 1000.0)
    print(f"replan {i}: {lat[-1]:.0f} ms")

print(f"replan mean (excl. first/JIT): {sum(lat[1:])/len(lat[1:]):.0f} ms")
