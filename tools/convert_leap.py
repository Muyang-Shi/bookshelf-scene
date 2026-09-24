"""One-off: convert the LEAP hand URDF (inputs/robots/leaphand_urdf) to a MuJoCo MJCF with position actuators.

Output: sam3d_scene/assets/leap_hand/leap_hand.xml (+ meshes/). Mount frame is the URDF "base" link.
"""
import os, re, shutil
import xml.etree.ElementTree as ET
import mujoco

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inputs", "robots", "leaphand_urdf")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "leap_hand")
KP, KV, FORCE = 3.0, 0.1, 0.95          # 0.95 Nm = URDF effort limit (Dynamixel XC330)
DAMPING, ARMATURE = 0.1, 0.001

os.makedirs(OUT, exist_ok=True)
shutil.copytree(os.path.join(SRC, "meshes"), os.path.join(OUT, "meshes"), dirs_exist_ok=True)
shutil.copy(os.path.join(SRC, "LICENSE.txt"), OUT)

urdf = open(os.path.join(SRC, "leaphand.urdf")).read()
ext = (f'<mujoco><compiler meshdir="{OUT}" strippath="false" discardvisual="false" '
       f'fusestatic="false" balanceinertia="true"/></mujoco>')
urdf = re.sub(r"(<robot[^>]*>)", r"\1" + ext, urdf, count=1)
m = mujoco.MjModel.from_xml_string(urdf)
tmp = os.path.join(OUT, "_raw.xml")
mujoco.mj_saveLastXML(tmp, m)

root = ET.parse(tmp).getroot()
os.remove(tmp)
root.set("model", "leap_hand")
comp = root.find("compiler")
comp.attrib = {"angle": "radian", "meshdir": "meshes/"}
for mesh in root.iter("mesh"):
    mesh.set("file", mesh.get("file").split("meshes/", 1)[-1])

default = ET.Element("default")
cls = ET.SubElement(default, "default", {"class": "leap"})
ET.SubElement(cls, "joint", {"damping": str(DAMPING), "armature": str(ARMATURE), "frictionloss": "0.001"})
ET.SubElement(cls, "position", {"kp": str(KP), "kv": str(KV), "forcerange": f"-{FORCE} {FORCE}"})
vis = ET.SubElement(cls, "default", {"class": "leap_visual"})
ET.SubElement(vis, "geom", {"contype": "0", "conaffinity": "0", "group": "2", "density": "0"})
col = ET.SubElement(cls, "default", {"class": "leap_collision"})
# stiff contacts (same as the scene's objects) so 0.95 Nm fingers can't sink into each other
ET.SubElement(col, "geom", {"group": "3", "friction": "1 0.01 0.001", "rgba": "0.2 0.6 0.9 0.4",
                            "solref": "0.004 1", "solimp": "0.95 0.99 0.001"})
root.insert(list(root).index(comp) + 1, default)

joints = []
for body in root.iter("body"):
    body.set("childclass", "leap")
    for g in body.findall("geom"):
        if g.get("type") == "mesh" and not g.get("mesh", "").startswith("white_tip"):
            for a in ("contype", "conaffinity", "group", "density"):
                g.attrib.pop(a, None)
            g.set("class", "leap_visual")
            g.set("rgba", "0.12 0.12 0.12 1")
        else:
            for a in ("contype", "conaffinity", "group"):
                g.attrib.pop(a, None)
            g.set("class", "leap_collision")
            if g.get("mesh", "").startswith("white_tip"):   # rubber tips: collide and render
                g.set("group", "2")
                g.set("rgba", "0.95 0.95 0.95 1")
                g.set("condim", "4")   # torsional friction for stable fingertip grasps
    for j in body.findall("joint"):
        for a in ("damping", "armature", "frictionloss"):
            j.attrib.pop(a, None)
        joints.append(j.get("name"))
top = root.find("worldbody").find("body")   # URDF "base" link = mount frame
top.set("name", "base")

# palm touches its direct children at rest; parent-child filtering doesn't apply if the palm is welded to world
contact = ET.SubElement(root, "contact")
palm = top.find("body")
for child in palm.findall("body"):
    ET.SubElement(contact, "exclude", {"body1": palm.get("name"), "body2": child.get("name")})

act = ET.SubElement(root, "actuator")
for jn in sorted(joints, key=int):
    lo, hi = m.jnt_range[m.joint(jn).id]
    ET.SubElement(act, "position", {"name": f"a{jn}", "joint": jn, "class": "leap",
                                    "ctrlrange": f"{lo:.4f} {hi:.4f}"})
ET.indent(root)
ET.ElementTree(root).write(os.path.join(OUT, "leap_hand.xml"))
print("joints", sorted(joints, key=int), "->", os.path.join(OUT, "leap_hand.xml"))
