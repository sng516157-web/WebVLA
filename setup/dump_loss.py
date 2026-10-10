#!/usr/bin/env python3
"""Dump training loss history from the offline wandb run's local datastore."""
import glob
import json

from wandb.sdk.internal.datastore import DataStore

run = sorted(glob.glob("/root/webvla/octo/wandb/offline-run-*libero_spatial_bs16_lr5e5*"))[-1]
ds = DataStore()
ds.open_for_scan(glob.glob(f"{run}/*.wandb")[0])

records = []
debug_shown = 0
while True:
    rec = ds.scan_record()
    if rec is None:
        break
    try:
        data = ds.scan_data()
    except AssertionError:
        continue
    if data is None:
        continue
    if debug_shown < 2:
        debug_shown += 1
        print("DEBUG record sample:", repr(data)[:300])
    try:
        d = json.loads(data)
    except Exception:
        continue
    if isinstance(d, dict) and "training/loss" in d:
        records.append((d.get("_step", 0), d["training/loss"]))

records.sort()
for step, loss in records[::5]:
    print(f"{step:6d} {loss:.4f}")
print(f"total logged points: {len(records)}")
if records:
    print(f"first: {records[0]}, last: {records[-1]}")
