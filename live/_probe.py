"""Probe the live-mode seams against the real LIBERO/robosuite install.

Prints, for one libero_spatial task:
  - camera id/fovy/xpos/xmat for agentview
  - movable object names, body ids, free-joint names + qvel addrs
  - main_table geometry for the support-plane z
  - raw agentview_image shape and where a known object lands under the two
    candidate pixel conventions (v-down vs v-up), sampled against pixel colors
"""
import numpy as np

from sim.wrapper import LiberoWrapper

w = LiberoWrapper("libero_spatial", render_size=256)
obs = w.reset(task_id=0, seed=0)

sim = w.env.sim
model, data = sim.model, sim.data

cam_id = model.camera_name2id("agentview")
print("cam_id:", cam_id)
print("cam_fovy:", float(model.cam_fovy[cam_id]))
print("cam_xpos:", data.cam_xpos[cam_id].tolist())
print("cam_xmat:", np.asarray(data.cam_xmat[cam_id]).tolist())

inner = w.env.env
print("n objects:", len(inner.objects))
for o in inner.objects:
    name = o.name
    body_id = inner.obj_body_id.get(name)
    joints = getattr(o, "joints", None)
    print(f"obj {name!r} body_id={body_id} pos={data.body_xpos[body_id].round(3).tolist()} joints={joints}")
    if joints:
        for j in joints:
            try:
                qpos_adr = model.get_joint_qpos_addr(j)
                qvel_adr = model.get_joint_qvel_addr(j)
                print(f"   joint {j!r} qposadr={qpos_adr} qveladr={qvel_adr}")
            except Exception as e:
                print(f"   joint {j!r} ERR {e}")

tid = model.body_name2id("table")
print("main_table body id:", tid, "pos:", data.body_xpos[tid].tolist())
raw_model = model._model
import mujoco as dm
for g in range(raw_model.ngeom):
    if raw_model.geom_bodyid[g] == tid:
        print("  geom", g, dm.mj_id2name(raw_model, dm.mjtObj.mjOBJ_GEOM, g),
              "type", int(raw_model.geom_type[g]), "size", raw_model.geom_size[g].tolist(),
              "pos", raw_model.geom_pos[g].tolist())

raw = w._raw["agentview_image"]
print("raw agentview_image shape:", raw.shape)
H, W = raw.shape[:2]
fovy = np.deg2rad(float(model.cam_fovy[cam_id]))
f = (H / 2.0) / np.tan(fovy / 2.0)
cam_pos = np.asarray(data.cam_xpos[cam_id])
cam_mat = np.asarray(data.cam_xmat[cam_id]).reshape(3, 3)


def project(p, v_down):
    d = np.asarray(p) - cam_pos
    pc = cam_mat.T @ d  # camera frame: x right, y up, z back
    u = f * pc[0] / (-pc[2]) + W / 2.0
    v = (-1.0 if v_down else 1.0) * f * pc[1] / (-pc[2]) + H / 2.0
    return u, v


def px(u, v):
    return raw[int(np.clip(round(v), 0, H - 1)), int(np.clip(round(u), 0, W - 1))].tolist()


for o in inner.objects:
    p = data.body_xpos[inner.obj_body_id[o.name]]
    ud, vd = project(p, True)
    uu, vu = project(p, False)
    print(f"{o.name!r} world={p.round(3).tolist()}")
    print(f"   v-down: ({ud:.1f},{vd:.1f}) rgb={px(ud, vd)}   v-up: ({uu:.1f},{vu:.1f}) rgb={px(uu, vu)}")

# also print the pixel colors of a grid to identify table color
print("corners:", px(0, 0), px(W - 1, 0), px(0, H - 1), px(W - 1, H - 1), "center:", px(W / 2, H / 2))
