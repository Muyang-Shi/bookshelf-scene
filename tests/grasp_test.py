"""Scripted LEAP-hand cup grasp to sanity-check the scene's contacts, friction and control.

Tests:
  table  - move cup0 onto the open table, wrap-grasp it, lift 15 cm, hold 2 s
  book   - stand one book on the table, spine toward the robot, grip it by the spine (palm on the
           spine, fingers and thumb on the covers), lift 15 cm, hold 2 s
  shelf  - wrap-grasp the middle cup (cup2) on the bottom board, lift 3 cm, pull it out
           (currently unreachable: no collision-free arm pose gets the hand in at cup height)

The arm follows IK waypoints (joint-space interpolation on the position servos) with gravity
compensation on the arm joints, as a real UR controller would do. The hand closes with position
targets past contact, so the 0.95 Nm servo limit sets the grip force.

Usage:
  python tests/grasp_test.py [table|book|shelf]           # headless, writes a GIF
  python tests/grasp_test.py [table|book|shelf] --view    # watch it live (on macOS use mjpython, not python)
"""
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "..", "scene.xml")

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# Hand closing targets (rad): index/middle/ring = (mcp side, mcp flex, pip, dip), thumb = 12..15.
# Targets sit past the contact point so the servos squeeze at their force limit.
CLOSE = {1: 1.4, 2: 1.2, 3: 0.9, 5: 1.4, 6: 1.2, 7: 0.9, 9: 1.4, 10: 1.2, 11: 0.9,
         12: 1.6, 13: 0.2, 14: 0.9, 15: 0.9}
# A 5 cm book is much thinner than a cup, so the thumb reaches CLOSE's targets without touching
# hard and stops pressing. Deeper thumb targets keep it squeezing the far cover.
CLOSE_BOOK = {**CLOSE, 12: 2.05, 14: 1.5, 15: 1.5}

# Grasp frame in tool0 coordinates: the cup axis sits 5 cm along the fingers (-tool y) and 8.5 cm in
# front of the palm (+tool z), finger stack centered on tool x = 0. Found by trial: further out along the
# fingers only the fingertips reach the cup and it slips; closer in, the finger motor blocks hit it.
CUP_IN_TOOL = np.array([0.0, -0.05, 0.085])
# Hand orientation per test (columns = tool0 x, y, z in world). Fingers run along -tool y, the palm
# faces +tool z, the thumb sits at -tool x. Thumb up in both.
TOOL_R = {
    # palm faces +X (away from the robot), fingers toward the viewer (-Y): approach along the reach
    "table": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),
    "book": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),     # same palm-forward grasp
    # palm faces +Y (into the shelf), fingers along +X: same as the 'ready' keyframe
    "shelf": np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float),
}


class Rig:
    def __init__(self, tool_r):
        self.R = tool_r
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.arm_q = [m.joint(f"ur5_{j}").qposadr[0] for j in JOINTS]
        self.arm_v = [m.joint(f"ur5_{j}").dofadr[0] for j in JOINTS]
        self.arm_a = [m.actuator(f"ur5_{a}").id for a in
                      ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]]
        self.hand_a = {int(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)[len("ur5_leap_a"):]): a
                       for a in range(m.nu) if "leap" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)}
        self.tool = m.body("ur5_tool0").id
        self.kin = mujoco.MjData(m)
        self.frames, self.viewer, self.renderer = [], None, None

    def ik(self, cup_pos, seed, ignore=()):
        """Collision-free arm joint angles putting the grasp frame on cup_pos.

        Tries the given seed first (continuity with the current pose), then standard UR postures and
        random seeds; keeps the collision-free solution closest to `seed`.
        """
        rng = np.random.default_rng(0)
        seeds = [seed, np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]),
                 np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, 0]),
                 np.array([0, -1.0, 1.8, -2.4, -np.pi / 2, 0])]
        seeds += [rng.uniform(-np.pi, np.pi, 6) for _ in range(40)]
        best, why = None, ""
        for s0 in seeds:
            try:
                q = self._ik(cup_pos, s0, ignore)
            except (AssertionError, RuntimeError) as e:
                why = str(e)
                continue
            q = seed + (q - seed + np.pi) % (2 * np.pi) - np.pi   # nearest equivalent to the current pose
            if np.all(np.abs(q[[2]]) <= np.pi) and (best is None or np.linalg.norm(q - seed) < np.linalg.norm(best - seed)):
                best = q
            if best is not None and s0 is seed:
                break
        if best is None:
            raise RuntimeError(f"no collision-free IK for {np.round(cup_pos, 3)} ({why})")
        return best

    def _ik(self, cup_pos, seed, ignore=()):
        """Single-seed damped least squares; refuses poses where the robot collides with itself or
        the scene (bodies in `ignore` excepted)."""
        m, k = self.m, self.kin
        k.qpos[:] = self.d.qpos
        k.qpos[self.arm_q] = seed
        tq = np.zeros(4)
        mujoco.mju_mat2Quat(tq, self.R.flatten())
        target = cup_pos - self.R @ CUP_IN_TOOL
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(200):
            mujoco.mj_kinematics(m, k)
            mujoco.mj_comPos(m, k)
            ep = target - k.xpos[self.tool]
            er = np.zeros(3)
            mujoco.mju_subQuat(er, tq, k.xquat[self.tool].copy())
            er = k.xmat[self.tool].reshape(3, 3) @ er
            e = np.r_[ep, er]
            if np.linalg.norm(e) < 1e-6:
                break
            mujoco.mj_jacBody(m, k, jp, jr, self.tool)
            J = np.r_[jp, jr][:, self.arm_v]
            k.qpos[self.arm_q] += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        q = (k.qpos[self.arm_q] + np.pi) % (2 * np.pi) - np.pi
        assert np.linalg.norm(e) < 1e-4, f"IK failed ({np.linalg.norm(e):.4f})"
        k.qpos[self.arm_q] = q
        mujoco.mj_forward(m, k)
        robot = m.body("ur5_pedestal").id
        skip = {m.body(b).id for b in ignore}
        for c in k.contact[:k.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if robot in (m.body_rootid[b1], m.body_rootid[b2]) and not {b1, b2} & skip and c.dist < -1e-3:
                raise RuntimeError(f"IK pose collides: {mujoco.mj_id2name(m, 1, b1)} / {mujoco.mj_id2name(m, 1, b2)}")
        return q

    def step(self, n=1):
        m, d = self.m, self.d
        for _ in range(n):
            d.qfrc_applied[self.arm_v] = d.qfrc_bias[self.arm_v]   # gravity compensation
            mujoco.mj_step(m, d)
            if self.viewer is not None and d.time % 0.02 < m.opt.timestep:
                self.viewer.sync()
                time.sleep(0.02)
            if self.renderer is not None and len(self.frames) < 400 and round(d.time / m.opt.timestep) % 25 == 0:
                self.renderer.update_scene(d, self.cam)
                self.frames.append(self.renderer.render().copy())

    def move(self, q_goal, secs):
        q0 = self.d.ctrl[self.arm_a].copy()
        n = int(secs / self.m.opt.timestep)
        for i in range(n):
            s = (i + 1) / n
            s = s * s * (3 - 2 * s)   # smoothstep
            self.d.ctrl[self.arm_a] = q0 + s * (q_goal - q0)
            self.step()

    def hand(self, targets, secs):
        a = [self.hand_a[j] for j in targets]
        c0 = self.d.ctrl[a].copy()
        c1 = np.array(list(targets.values()))
        n = int(secs / self.m.opt.timestep)
        for i in range(n):
            self.d.ctrl[a] = c0 + (i + 1) / n * (c1 - c0)
            self.step()

    def cup_contacts(self, cup):
        b = self.m.body(cup).id
        hand = set()
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if b in (b1, b2):
                other = mujoco.mj_id2name(self.m, 1, b2 if b1 == b else b1)
                if other.startswith("ur5_leap"):
                    hand.add(other[len("ur5_leap_"):])
        return sorted(hand)


def run(test, view=False):
    r = Rig(TOOL_R[test])
    m, d = r.m, r.d
    mujoco.mj_resetDataKeyframe(m, d, m.key("ready").id)
    cup = {"table": "cup0", "book": "book03_book3", "shelf": "cup2"}[test]   # the object to grasp
    top = m.geom("table_top")
    table_h = top.pos[2] + top.size[2]
    adr = m.jnt_qposadr[m.body(cup).jntadr[0]]
    if test == "table":   # put cup0 on the open table in front of the shelf
        d.qpos[adr:adr + 7] = [0.0, -0.12, table_h + 0.002, 1, 0, 0, 0]
    elif test == "book":  # stand the book on the table, yawed -90 deg so its spine faces the robot (-X)
        d.qpos[adr:adr + 7] = [0.10, -0.12, table_h + 0.002, np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)]
    mujoco.mj_forward(m, d)

    if view:
        r.viewer = mujoco.viewer.launch_passive(m, d)
    else:
        r.renderer = mujoco.Renderer(m, 360, 480)
        r.cam = mujoco.MjvCamera()
        r.cam.azimuth, r.cam.elevation, r.cam.distance = (75, -12, 0.9) if test == "shelf" else (60, -20, 1.2)
        r.cam.lookat[:] = d.xpos[m.body(cup).id] + [0, 0, 0.08]

    others = [b for b in range(m.nbody) if (mujoco.mj_id2name(m, 1, b) or "").startswith(("cup", "book"))
              and mujoco.mj_id2name(m, 1, b) != cup]
    r.step(250)                                   # settle
    others0 = {b: d.xpos[b].copy() for b in others}
    c0 = d.xpos[m.body(cup).id].copy()
    if test == "book":
        # palm on the spine, grip point 7.5 cm in from the spine face (close to the book's centre
        # of mass, 9 cm in, so it doesn't swing out of the pinch), centered on the thickness, 16 cm up
        g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(cup).id and m.geom_contype[g]][0]
        md = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[md]:m.mesh_vertadr[md] + m.mesh_vertnum[md]]
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        grip = np.array([w[:, 0].min() + 0.075, (w[:, 1].min() + w[:, 1].max()) / 2, table_h + 0.16])
    else:
        grip = c0 + [0, 0, 0.085]   # finger stack (9 cm) spans 4-13 cm up the 12.9 cm cup, clear of the table
    n = r.R[:, 2]               # palm normal = approach direction
    seed = d.qpos[r.arm_q].copy()

    # Waypoints (cup-axis positions): come in high so the forearm never sweeps through the cup,
    # drop to grip height 12 cm in front of it, then slide straight in along +Y.
    if test in ("table", "book"):
        via = [grip - 0.14 * n + [0, 0, 0.20], grip - 0.14 * n]
    else:                                          # shelf: middle board above limits the height
        via = [grip - 0.22 * n + [0, 0, 0.05], grip - 0.14 * n]
    q = seed
    for p in via:
        q = r.ik(p, q)
        r.move(q, 1.5)
    q_in = r.ik(grip, q, ignore=[cup])
    r.move(q_in, 1.5)
    r.hand(CLOSE_BOOK if test == "book" else CLOSE, 1.0)
    r.step(250)
    touching = r.cup_contacts(cup)
    ob = m.body(cup).id

    def in_hand():   # object position in the tool frame
        return d.xmat[r.tool].reshape(3, 3).T @ (d.xpos[ob] - d.xpos[r.tool])
    grasped_at = in_hand()
    z_grasped = d.xpos[ob][2]

    lift = [0, 0, 0.03] if test == "shelf" else [0, 0, 0.15]
    q_up = r.ik(grip + lift, q_in, ignore=[cup])
    r.move(q_up, 1.5)
    if test == "shelf":
        q_out = r.ik(grip + lift + [0, -0.25, 0], q_up, ignore=[cup])   # pull straight out of the shelf
        r.move(q_out, 2.0)
    r.step(1000)                                   # hold 2 s

    c1 = d.xpos[ob]
    held = r.cup_contacts(cup)
    slip = np.linalg.norm(in_hand() - grasped_at)          # how far it moved in the hand while carried
    lifted = c1[2] - z_grasped
    tilt = np.degrees(np.arccos(np.clip(d.xmat[ob][8], -1, 1)))
    moved = max((np.linalg.norm(d.xpos[b] - p) for b, p in others0.items()), default=0)
    unstable = bool(np.isnan(d.qpos).any() or any(w.number for w in d.warning))
    ok = len(held) >= 2 and lifted > 0.8 * lift[2] and slip < 0.03 and not unstable
    print(f"[{test}] {cup}: start {np.round(c0, 3)} -> end {np.round(c1, 3)}; lifted {lifted * 100:.1f} cm "
          f"of {lift[2] * 100:.0f}, slip in hand {slip * 1000:.0f} mm, tilt {tilt:.1f} deg")
    print(f"  hand links touching {cup} after closing: {touching}")
    print(f"  hand links touching {cup} at the end:    {held}")
    print(f"  other cups/books displaced: max {moved * 1000:.0f} mm   unstable: {unstable}")
    if ok:
        verdict = f"PASS - {cup} grasped and carried"
    elif len(held) >= 2 and lifted > 0.5 * lift[2] and not unstable:
        verdict = f"PARTIAL - {cup} lifted {lifted * 100:.0f} cm but slipped {slip * 1000:.0f} mm in the hand"
    else:
        verdict = "FAIL"
    print(f"  RESULT: {verdict}")

    if r.frames:
        from PIL import Image
        out = os.path.join(HERE, f"grasp_{test}.gif")
        imgs = [Image.fromarray(f) for f in r.frames[::2]]
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=50, loop=0)
        print(f"  animation: {out}")
    if r.viewer is not None:
        while r.viewer.is_running():
            r.step()
    return ok


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tests = args or ["table"]
    for t in tests:
        try:
            run(t, view="--view" in sys.argv)
        except RuntimeError as e:
            print(f"[{t}] could not plan the grasp: {e}")
