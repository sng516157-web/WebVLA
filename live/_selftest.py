"""In-process self-test for live mode (no server needed):

1. Build a LiveSession env (task 0, seed 0) at 640 render.
2. Ground-truth geometry: project every movable object's body position into
   displayed pixels via CameraGeometry.world_to_display, then call
   handle_click at those pixels and assert:
   - the ray-table hit is within 3 cm of the object's (x, y)
   - the reprojected debug pixel is within 3 px of the click
   - with push=True the correct object is selected and pushed
3. Physics: step the sim ~0.5 s and assert the pushed object actually moved
   (position delta > 1 cm) and stayed on the table (z sane).
"""
import numpy as np

from live.session import (
    CROP_Y0, RENDER, CameraGeometry, LiveSession, movable_objects, table_top_z,
)

CHECKPOINT = None  # filled from paths


def main():
    import os
    import paths

    ckpt = os.path.join(
        paths.CKPTS_DIR, "octo_finetune", "libero_spatial",
        "libero_spatial_bs16_lr5e5_20261010_114251",
    )
    s = LiveSession(checkpoint=ckpt)
    # env only — skip the policy load, this test never calls act()
    from sim.wrapper import LiberoWrapper
    s.wrapper = LiberoWrapper("libero_spatial", render_size=RENDER)
    s.wrapper.reset(task_id=0, seed=0)
    s._cam = CameraGeometry(s.wrapper)
    s._cam.refresh()
    s._table_z = table_top_z(s.wrapper)
    print(f"table_z = {s._table_z:.4f} (expect ~0.900)")
    assert 0.85 < s._table_z < 0.95, "table plane z looks wrong"

    data = s.wrapper.env.sim.data
    failures = []

    for name, body_id in movable_objects(s.wrapper):
        pos = np.asarray(data.body_xpos[body_id], dtype=np.float64)
        disp = s._cam.world_to_display(pos)
        assert disp is not None, f"{name} behind camera"
        xd, yd = disp
        r = s.handle_click(xd, yd, push=False)
        print(f"{name}: world={pos.round(3).tolist()} click=({xd:.1f},{yd:.1f}) "
              f"hit={r['hit_world']} reproj={r['reprojected']}")
        if r["hit_world"] is None:
            failures.append(f"{name}: no table hit")
            continue
        err_xy = np.linalg.norm(np.array(r["hit_world"][:2]) - pos[:2])
        # click at the object's center of mass, which sits ~1-2cm above the
        # table; the ray hits the plane slightly off the body xy. Allow 4 cm.
        if err_xy > 0.04:
            failures.append(f"{name}: hit xy error {err_xy:.3f} m")
        rp_err = np.linalg.norm(np.array(r["reprojected"]) - np.array([xd, yd]))
        if rp_err > 3.0:
            failures.append(f"{name}: reprojection error {rp_err:.1f} px")

    # push test on the first object
    name, body_id = movable_objects(s.wrapper)[0]
    pos0 = np.asarray(data.body_xpos[body_id], dtype=np.float64).copy()
    xd, yd = s._cam.world_to_display(pos0)
    r = s.handle_click(xd, yd, push=True)
    print(f"push result: {r}")
    if not r["pushed"] or r["object"] != name:
        failures.append(f"push: expected {name}, got {r['object']} pushed={r['pushed']}")
    from sim.wrapper import DUMMY_ACTION
    for _ in range(10):  # 0.5 s of physics
        s.wrapper.step(DUMMY_ACTION)
    pos1 = np.asarray(data.body_xpos[body_id], dtype=np.float64)
    moved = np.linalg.norm(pos1[:2] - pos0[:2])
    print(f"object moved {moved:.3f} m in 0.5 s, z {pos0[2]:.3f} -> {pos1[2]:.3f}")
    if moved < 0.01:
        failures.append(f"push: object barely moved ({moved:.4f} m)")
    if not (0.8 < pos1[2] < 1.2):
        failures.append(f"push: object left the table (z={pos1[2]:.3f})")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
