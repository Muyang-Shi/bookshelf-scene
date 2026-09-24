"""Open scene.xml in the MuJoCo viewer, starting from a keyframe (default: home).

Usage:  python view.py [home|ready]
"""
import os
import sys
import mujoco
import mujoco.viewer

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene.xml")
key = sys.argv[1] if len(sys.argv) > 1 else "home"
m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
mujoco.viewer.launch(m, d)
