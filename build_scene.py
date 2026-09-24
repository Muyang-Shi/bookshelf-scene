"""Build the bookshelf MuJoCo scene (table + shelf + books + cups + UR5 with LEAP hand) from SAM3D meshes.

SAM3D exports every object normalized to a unit bounding box, glTF Y-up. This script
  1. rotates each mesh to MuJoCo Z-up (glTF +Z "front" -> world -Y),
  2. applies an optional yaw, then scales it to the real-world size in ASSETS,
  3. recenters it so the origin is the bottom-center of its bounding box,
  4. writes textured OBJ + PNG per asset and a scene.xml that places everything.

World frame: origin under the table center on the floor, +Y toward the shelf (back),
+X to the right when facing the shelf, +Z up.

Real sizes/poses come from a metric phone LiDAR capture: `python tools/measure_scan.py capture.zip
--write` measures it and writes scan_measurements.json, which overrides the defaults below. The
defaults are the 24 Sep 2026 capture (scene_with_obj.zip). Cup/book sizes follow from the shelf
compartments and the reference photo.

Usage:  python build_scene.py   then   python view.py
"""
import json
import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import mujoco

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# Inputs live inside the package so it builds on any machine. SAM3D_GLB_DIR overrides the mesh folder.
GLB_DIR = os.environ.get("SAM3D_GLB_DIR", os.path.join(OUT_DIR, "inputs", "sam3d_glb"))
UR5_SRC = os.path.join(OUT_DIR, "inputs", "robots", "universal_robots_ur5", "ur5.xml")
SCAN_JSON = os.path.join(OUT_DIR, "scan_measurements.json")   # from tools/measure_scan.py --write
UR5_XML = os.path.join(OUT_DIR, "assets", "ur5", "ur5_leap.xml")      # generated: UR5 + LEAP on tool0
LEAP_XML = os.path.join(OUT_DIR, "assets", "leap_hand", "leap_hand.xml")  # from tools/convert_leap.py
# LEAP "base" frame relative to UR5 tool0 (adapter offset). Adjust to match the real mount.
HAND_MOUNT_POS = (0.0, 0.0, 0.0)
HAND_MOUNT_EULER = (0.0, 0.0, 0.0)   # radians, xyz

# size: target world extents (x, y, z) in metres; None entries are filled by uniform scaling
#       from the non-None ones. yaw: degrees about world Z, applied before scaling
#       (books: yaw puts the spine toward -Y, i.e. facing out of the shelf).
ASSETS = {
    "table": dict(size=(0.77, 0.98, 0.753), yaw=90),
    "shelf": dict(size=(0.654, 0.22, 0.677), yaw=0),
    "cup":   dict(size=(0.086, None, None), yaw=0),
    "book0": dict(size=None, yaw=90),   # height set from BOOK_HEIGHT_FRACTIONS below
    "book1": dict(size=None, yaw=90),   # height set from BOOK_HEIGHT_FRACTIONS below
    "book2": dict(size=None, yaw=90),   # height set from BOOK_HEIGHT_FRACTIONS below
    "book3": dict(size=None, yaw=180),   # height set from BOOK_HEIGHT_FRACTIONS below
}

# Left-to-right order on the middle board. Each mesh is one physical book series, so same-label
# books sit together and share a height: book3 = the four tall ones on the left, book1 = the medium
# four (VOGUE cover), book0 = the short pair (Glace cover, dark spines), book2 = the tall pair on the right.
BOOK_ORDER = ["book3", "book3", "book3", "book3", "book1", "book1",
              "book1", "book1", "book0", "book0", "book2", "book2"]
# Spine height of each book series (mesh) as a fraction of the clear compartment height. Each is
# the mean of that series' spines in the reference photo (perspective-corrected), whose readings were
#   book3: STYLE .965, WE ALL SAID .968, NORDIC HOUSEHOLD .970, SUPERIOR .980
#   book1: CELINE .874, RADIO ASTRONOMER .870, LOEWE .875, VOGUE .873
#   book0: CONSIDERED .788, Glace .800        book2: PROJECT 82 .930, BETTER LIFE .935
# The height belongs to the mesh, so every copy of a series is the same size.
BOOK_HEIGHT_FRACTIONS = {"book3": 0.971, "book1": 0.873, "book0": 0.794, "book2": 0.933}
BOOK_GAP = 0.002

SHELF_X = 0.02            # shelf center offset from the table centerline
SHELF_BACK_GAP = 0.003    # table back edge to shelf back
# UR5 mount (top of pedestal) in world frame. Height = lowest blue (shoulder cap) in the scan minus
# its 0.024 m offset in the model. Base yaw 0: the shoulder-pan joint handles the facing.
UR5_MOUNT = (-0.58, 0.0, 0.551)

if os.path.exists(SCAN_JSON):   # measured numbers from a capture override the defaults above
    with open(SCAN_JSON) as f:
        scan = json.load(f)
    ASSETS["table"]["size"] = (scan["table_w"], scan["table_d"], scan["table_h"])
    ASSETS["shelf"]["size"] = (scan["shelf_w"], scan["shelf_d"], scan["shelf_h"])
    SHELF_X, SHELF_BACK_GAP = scan["shelf_x"], max(scan["shelf_back_gap"], 0.0)
    if "mount_z" in scan:
        UR5_MOUNT = (scan["mount_x"], scan["mount_y"], scan["mount_z"])
    print(f"using scan measurements from {os.path.basename(SCAN_JSON)} ({', '.join(scan.get('sources', []))})")

TABLE_H = ASSETS["table"]["size"][2]
TABLE_D = ASSETS["table"]["size"][1]
SHELF_W, SHELF_D, SHELF_H = ASSETS["shelf"]["size"]
SHELF_Y = TABLE_D / 2 - SHELF_D / 2 - SHELF_BACK_GAP
PANEL_T = 0.009                                # shelf board half-thickness
# Clear height of the book compartment: middle-board top to upper-board underside
# (boards at normalized shelf heights -0.05 and 0.36, see BOTTOM_BOARD_TOP below).
BOOK_CLEAR = (0.36 - (-0.05)) * SHELF_H - 2 * PANEL_T
for _n, _f in BOOK_HEIGHT_FRACTIONS.items():
    assert _f < 1, f"{_n}: height fraction {_f} doesn't fit the compartment"
    ASSETS[_n]["size"] = (None, None, _f * BOOK_CLEAR)
SHELF_INNER = SHELF_W - 4 * PANEL_T

# Shelf board levels in the normalized SAM3D mesh (glTF y in [-0.5, 0.5]), measured from shelf.glb.
BOTTOM_BOARD_TOP = -0.48
MIDDLE_BOARD_TOP = -0.04

N_CUPS = 6
CUP_MASS, BOOK_MASS = 0.05, 0.3

PEDESTAL_R = 0.07
# Standard UR home (upper arm up, forearm level, tool down); pan 0 faces the table (+X).
HOME_Q = (0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0)
# Ready pose: tool0 in front of the book compartment, tool z-axis pointing into the shelf (+Y).
READY_TOOL_POS = (SHELF_X, SHELF_Y - 0.30, TABLE_H + 0.35)
# IK start points, tried in order; the first converged, collision-free solution wins
# (a single seed can land on an elbow-down branch that drives the forearm into the table).
READY_SEEDS = [(-2.94, -2.11, -1.02, 0.0, 2.94, 1.57),   # elbow-up, reaching over the table
               (np.pi, -1.9, 1.9, -1.57, -1.57, 0),
               (0.0, -1.2, -1.8, 0.0, 1.57, 0),
               (0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0)]


def straighten(mesh, name, max_deg=15):
    """Remove a small tilt so the mesh's tight (oriented) bounding box lines up with the axes.

    SAM3D books can lean a few degrees; several identical leaning copies side by side then tip
    against each other. Only the small residual rotation is applied, so the facing is unchanged.
    """
    to_obb, _ = trimesh.bounds.oriented_bounds(mesh)
    R = to_obb[:3, :3]
    S = np.zeros((3, 3))                       # nearest signed permutation = the big part of R
    for i in range(3):
        j = int(np.argmax(np.abs(R[i])))
        S[i, j] = np.sign(R[i, j])
    small = S.T @ R                            # residual tilt, close to identity
    angle = np.degrees(np.arccos(np.clip((np.trace(small) - 1) / 2, -1, 1)))
    if abs(np.linalg.det(S) - 1) > 1e-6 or angle > max_deg:
        return
    T = np.eye(4)
    T[:3, :3] = small
    c = mesh.bounds.mean(0)
    mesh.apply_transform(trimesh.transformations.translation_matrix(c) @ T
                         @ trimesh.transformations.translation_matrix(-c))
    print(f"{name:6s} straightened by {angle:.1f} deg")


def load(name, cfg):
    mesh = trimesh.load(os.path.join(GLB_DIR, f"{name}.glb")).to_geometry()
    # glTF Y-up -> Z-up: (x, y, z) -> (x, -z, y)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    if name.startswith("book"):
        straighten(mesh, name)
    if cfg["yaw"]:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(cfg["yaw"]), [0, 0, 1]))
    ext = mesh.extents
    known = [t / e for t, e in zip(cfg["size"], ext) if t is not None]
    mesh.apply_scale([t / e if t is not None else np.mean(known) for t, e in zip(cfg["size"], ext)])
    return mesh


def export(name, mesh):
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    d = os.path.join(OUT_DIR, "assets", name)
    os.makedirs(d, exist_ok=True)
    mat = mesh.visual.material
    if hasattr(mat, "to_simple"):
        mesh.visual.material = mat.to_simple()
    mesh.visual.material.name = name
    mesh.visual.material.image.convert("RGB").save(os.path.join(d, f"{name}.png"))
    # include_texture=True is what writes the vt (UV) lines; MuJoCo ignores the .mtl itself
    obj, files = trimesh.exchange.obj.export_obj(
        mesh, include_texture=True, include_normals=True, return_texture=True, mtl_name=f"{name}.mtl")
    with open(os.path.join(d, f"{name}.obj"), "w") as f:
        f.write(obj)
    for fn, data in files.items():
        if fn.endswith(".mtl"):
            with open(os.path.join(d, fn), "wb") as f:
                f.write(data)
    print(f"{name:6s} extents={np.round(mesh.extents, 3)}")


def box(name, center, half, rgba="0.6 0.4 0.2 1"):
    c = " ".join(f"{v:.4f}" for v in center)
    h = " ".join(f"{v:.4f}" for v in half)
    return f'      <geom name="{name}" class="static_col" type="box" pos="{c}" size="{h}" rgba="{rgba}"/>'


def free_body(name, mesh, pos, mass, material=None):
    return [f'    <body name="{name}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}">',
            '      <freejoint/>',
            f'      <geom class="visual" mesh="{mesh}" material="{material or mesh}_mat"/>',
            f'      <geom class="obj_col" mesh="{mesh}" mass="{mass}"/>',
            "    </body>"]


def make_ur5_with_hand():
    """Copy the stock UR5 MJCF next to the scene and attach the LEAP hand at tool0."""
    d = os.path.dirname(UR5_XML)
    shutil.copytree(os.path.join(os.path.dirname(UR5_SRC), "assets"), os.path.join(d, "assets"), dirs_exist_ok=True)
    root = ET.parse(UR5_SRC).getroot()
    root.set("model", "ur5_leap")
    for tag in ("size", "keyframe"):   # the scene defines its own keyframe for the combined model
        for el in root.findall(tag):
            root.remove(el)
    ET.SubElement(root.find("asset"), "model", {"name": "leap", "file": os.path.relpath(LEAP_XML, d)})
    tool0 = next(b for b in root.iter("body") if b.get("name") == "tool0")
    frame = ET.SubElement(tool0, "frame", {"name": "leap_mount",
                                           "pos": " ".join(map(str, HAND_MOUNT_POS)),
                                           "euler": " ".join(map(str, HAND_MOUNT_EULER))})
    ET.SubElement(frame, "attach", {"model": "leap", "body": "base", "prefix": "leap_"})
    ET.indent(root)
    ET.ElementTree(root).write(UR5_XML)


def solve_ready_pose(xml):
    """Damped least-squares IK for the UR5 'ready' keyframe, tried from several seeds.

    Returns the home and ready qpos; the ready pose is the first seed that converges with the
    robot touching nothing but itself.
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    arm = [m.joint(f"ur5_{j}").qposadr[0] for j in joints]
    dof = [m.jnt_dofadr[m.joint(f"ur5_{j}").id] for j in joints]
    tool = m.body("ur5_tool0").id
    robot_root = m.body("ur5_pedestal").id
    # tool z -> +Y (into the shelf), tool x -> -Z
    target_mat = np.array([[0, 0, 0], [0, 0, 1], [-1, 0, 0]], float)
    target_mat[:, 1] = np.cross(target_mat[:, 2], target_mat[:, 0])
    target_quat = np.zeros(4)
    mujoco.mju_mat2Quat(target_quat, target_mat.flatten())
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def robot_hits_scene():
        for c in d.contact[:d.ncon]:
            roots = {m.body_rootid[m.geom_bodyid[c.geom1]], m.body_rootid[m.geom_bodyid[c.geom2]]}
            if robot_root in roots and len(roots) == 2:
                return True
        return False

    home = d.qpos.copy()
    home[arm] = HOME_Q
    for k, seed in enumerate(READY_SEEDS):
        d.qpos[:] = home
        d.qpos[arm] = seed
        for _ in range(300):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err_p = np.array(READY_TOOL_POS) - d.xpos[tool]
            err_r = np.zeros(3)
            mujoco.mju_subQuat(err_r, target_quat, d.xquat[tool].copy())
            err_r = d.xmat[tool].reshape(3, 3) @ err_r  # local -> world, to match jacr
            err = np.r_[err_p, err_r]
            if np.linalg.norm(err) < 1e-5:
                break
            mujoco.mj_jacBody(m, d, jacp, jacr, tool)
            J = np.r_[jacp, jacr][:, dof]
            d.qpos[arm] += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        # wrap to (-pi, pi] (same pose for revolute joints) so targets sit inside the actuator ctrlrange
        d.qpos[arm] = (d.qpos[arm] + np.pi) % (2 * np.pi) - np.pi
        mujoco.mj_forward(m, d)
        ok = np.linalg.norm(err) < 1e-4 and not robot_hits_scene()
        print(f"ready-pose IK seed {k}: pos {np.linalg.norm(err_p) * 1000:.2f} mm, "
              f"rot {np.degrees(np.linalg.norm(err_r)):.2f} deg, {'ok' if ok else 'rejected'}")
        if ok:
            return [("home", home), ("ready", d.qpos.copy())], m, arm
    raise RuntimeError("no collision-free IK solution for READY_TOOL_POS; move the target or add a seed")


def keyframe_ctrl(m, arm, qpos):
    ctrl = np.zeros(m.nu)   # hand actuators at 0 = open hand
    for a in range(m.nu):   # arm actuators hold the keyframe's joint angles
        adr = m.jnt_qposadr[m.actuator_trnid[a, 0]]
        if adr in arm:
            ctrl[a] = qpos[adr]
    return ctrl


def main():
    make_ur5_with_hand()
    meshes = {n: load(n, c) for n, c in ASSETS.items()}

    # fit book thickness (world X) so BOOK_ORDER exactly fills the compartment, as in the photo
    native = sum(meshes[n].extents[0] for n in BOOK_ORDER)
    avail = SHELF_INNER - BOOK_GAP * (len(BOOK_ORDER) + 1)
    thick = avail / native
    print(f"books: native row width {native:.3f} m, available {avail:.3f} m -> thickness x{thick:.3f}")
    for n in set(BOOK_ORDER):
        meshes[n].apply_scale([thick, 1, 1])
    for n, mesh in meshes.items():
        export(n, mesh)

    assets, body_xml = [], []
    for n in ASSETS:
        assets.append(f'    <texture name="{n}_tex" type="2d" file="assets/{n}/{n}.png"/>')
        assets.append(f'    <material name="{n}_mat" texture="{n}_tex" specular="0.2" shininess="0.1"/>')
        assets.append(f'    <mesh name="{n}" file="assets/{n}/{n}.obj"/>')
    assets.append(f'    <model name="ur5" file="{os.path.relpath(UR5_XML, OUT_DIR)}"/>')

    # --- UR5 on its pedestal (listed first so the arm joints lead qpos)
    mx, my, mz = UR5_MOUNT
    body_xml += [f'    <body name="ur5_pedestal" pos="{mx} {my} 0">',
                 f'      <geom name="pedestal_foot" type="cylinder" size="0.18 0.015" pos="0 0 0.015" rgba="0.1 0.1 0.1 1"/>',
                 f'      <geom name="pedestal_column" type="cylinder" size="{PEDESTAL_R} {mz / 2:.4f}" pos="0 0 {mz / 2:.4f}" rgba="0.15 0.15 0.15 1"/>',
                 f'      <frame pos="0 0 {mz}">',
                 '        <attach model="ur5" body="base_link" prefix="ur5_"/>',
                 '      </frame>',
                 '    </body>']

    # --- table: visual mesh + box collisions (top slab + 4 legs)
    tw, td, th = ASSETS["table"]["size"]
    body_xml += ['    <body name="table" pos="0 0 0">',
                 '      <geom class="visual" mesh="table" material="table_mat"/>',
                 box("table_top", (0, 0, th - 0.015), (tw / 2, td / 2, 0.015))]
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        body_xml.append(box(f"table_leg{i}", (sx * tw * 0.42, sy * td * 0.425, (th - 0.03) / 2),
                            (0.025, 0.025, (th - 0.03) / 2)))
    body_xml.append("    </body>")

    # --- shelf: visual mesh + panel boxes (sides, boards, back)
    def z(ny):  # normalized glTF y -> height in shelf frame
        return (ny + 0.5) * SHELF_H

    w2, d2, t = SHELF_W / 2, SHELF_D / 2, PANEL_T
    body_xml += [f'    <body name="shelf" pos="{SHELF_X} {SHELF_Y:.4f} {TABLE_H:.4f}">',
                 '      <geom class="visual" mesh="shelf" material="shelf_mat"/>',
                 box("shelf_left", (-w2 + t, 0, SHELF_H / 2), (t, d2, SHELF_H / 2)),
                 box("shelf_right", (w2 - t, 0, SHELF_H / 2), (t, d2, SHELF_H / 2))]
    for nm, ny in [("bottom", -0.49), ("middle", -0.05), ("upper", 0.36)]:
        body_xml.append(box(f"shelf_{nm}", (0, 0, z(ny)), (w2, d2, t)))
    for nm, (y0, y1) in [("back_low", (-0.5, -0.27)), ("back_high", (-0.07, 0.5))]:
        body_xml.append(box(f"shelf_{nm}", (0, d2 - 0.005, (z(y0) + z(y1)) / 2),
                            (w2, 0.005, (z(y1) - z(y0)) / 2)))
    body_xml.append("    </body>")

    # --- cups: one row on the bottom board
    pitch = SHELF_INNER / N_CUPS
    z_bottom = TABLE_H + z(BOTTOM_BOARD_TOP) + 0.002
    for i in range(N_CUPS):
        x = SHELF_X - SHELF_INNER / 2 + pitch * (i + 0.5)
        body_xml += free_body(f"cup{i}", "cup", (x, SHELF_Y - 0.01, z_bottom), CUP_MASS)

    # --- books: standing on the middle board, packed left to right
    z_mid = TABLE_H + z(MIDDLE_BOARD_TOP) + 0.002
    x = SHELF_X - SHELF_INNER / 2 + BOOK_GAP
    print(f"books: compartment clear height {BOOK_CLEAR:.3f} m, heights "
          + ", ".join(f"{n} {meshes[n].extents[2]:.3f}" for n in sorted(BOOK_HEIGHT_FRACTIONS)))
    for i, n in enumerate(BOOK_ORDER):
        wx = meshes[n].extents[0]
        body_xml += free_body(f"book{i:02d}_{n}", n, (x + wx / 2, SHELF_Y, z_mid), BOOK_MASS)
        x += wx + BOOK_GAP

    def scene(keyframe=""):
        return f"""<mujoco model="sam3d_shelf_scene_ur5">
  <compiler angle="radian" meshdir="./" texturedir="./"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <size nkey="2"/>
  <visual>
    <global azimuth="60" elevation="-25" offwidth="1920" offheight="1080"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0 0 0"/>
  </visual>
  <statistic extent="1.6" center="-0.2 0.1 0.9"/>
  <default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2"/>
    </default>
    <default class="obj_col">
      <geom type="mesh" group="3" friction="0.8 0.01 0.001" solref="0.004 1" rgba="0.8 0.3 0.3 0.5"/>
    </default>
    <default class="static_col">
      <geom group="3" friction="0.8 0.01 0.001"/>
    </default>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.35 0.37 0.38" rgb2="0.3 0.32 0.33" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
{chr(10).join(assets)}
  </asset>
  <worldbody>
    <light pos="0 -1 3" dir="0 0.3 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
{chr(10).join(body_xml)}
  </worldbody>
  <contact>
    <!-- the arm is welded to the world via the pedestal, so parent-child filtering doesn't apply here -->
    <exclude body1="ur5_base_link_inertia" body2="ur5_shoulder_link"/>
    <exclude body1="ur5_pedestal" body2="ur5_shoulder_link"/>
  </contact>
{keyframe}</mujoco>
"""

    # compile once (from OUT_DIR so relative asset paths resolve) to solve the ready pose
    cwd = os.getcwd()
    os.chdir(OUT_DIR)
    try:
        keys, m, arm = solve_ready_pose(scene())
    finally:
        os.chdir(cwd)
    fmt = lambda v: " ".join(f"{x:.6g}" for x in v)
    key = "  <keyframe>\n" + "".join(
        f'    <key name="{n}" qpos="{fmt(q)}" ctrl="{fmt(keyframe_ctrl(m, arm, q))}"/>\n' for n, q in keys
    ) + "  </keyframe>\n"
    with open(os.path.join(OUT_DIR, "scene.xml"), "w") as f:
        f.write(scene(key))
    print("wrote", os.path.join(OUT_DIR, "scene.xml"))


if __name__ == "__main__":
    main()
