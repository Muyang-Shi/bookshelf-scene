"""Inspect one textured asset, without loading the bookshelf or running dynamics.

Usage: python tools/view_asset.py --asset pen
Drag to orbit; scroll to zoom. Tab shows the standard MuJoCo controls.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def select_renderer():
    """Probe in child processes so a failed GLX library never loads here."""
    if not sys.platform.startswith("linux"):
        return
    probe = """import glfw
import sys
if not glfw.init():
    sys.exit(1)
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(64, 64, 'probe', None, None)
ok = bool(window)
if window:
    glfw.destroy_window(window)
glfw.terminate()
sys.exit(0 if ok else 1)
"""
    software = {"LIBGL_ALWAYS_SOFTWARE": "1", "__GLX_VENDOR_LIBRARY_NAME": "mesa"}
    for settings in ({}, software):
        try:
            result = subprocess.run([sys.executable, "-c", probe],
                                    env={**os.environ, **settings},
                                    capture_output=True, timeout=10)
        except subprocess.TimeoutExpired:
            continue
        if result.returncode == 0:
            os.environ.update(settings)
            if settings:
                print("Default OpenGL unavailable; using CPU rendering.", flush=True)
            return
    raise RuntimeError("Cannot create an OpenGL window. Check the desktop display and graphics driver.")


def load_asset(name):
    import mujoco
    import trimesh

    folder = ROOT / "assets" / name
    mesh_path, texture_path = folder / f"{name}.obj", folder / f"{name}.png"
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    cap_path = folder / "pen_cap.obj"
    has_cap = name == "pen" and cap_path.is_file()
    if has_cap:
        mesh = trimesh.util.concatenate([mesh, trimesh.load(cap_path, force="mesh", process=False)])
    center = mesh.bounds.mean(axis=0)
    extent = float(max(mesh.extents))
    root = ET.Element("mujoco", model=name)
    ET.SubElement(root, "statistic", center=" ".join(map(str, center)), extent=str(extent * 1.25))
    visual = ET.SubElement(root, "visual")
    elevation = -35 if mesh.extents[2] < 0.4 * max(mesh.extents[:2]) else -10
    ET.SubElement(visual, "global", azimuth="60", elevation=str(elevation))
    ET.SubElement(visual, "headlight", ambient="0.6 0.6 0.6", diffuse="0.5 0.5 0.5",
                  specular="0.65 0.65 0.65" if has_cap else "0 0 0")
    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "texture", type="skybox", builtin="flat", rgb1="0.88 0.90 0.92",
                  rgb2="0.88 0.90 0.92", width="128", height="128")
    ET.SubElement(assets, "texture", name="texture", type="2d", file=str(texture_path))
    ET.SubElement(assets, "material", name="material", texture="texture", specular="0.1", shininess="0.1")
    ET.SubElement(assets, "mesh", name="mesh", file=str(mesh_path))
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name=name, type="mesh", mesh="mesh", material="material",
                  contype="0", conaffinity="0")
    if has_cap:
        ET.SubElement(assets, "material", name="cap_material", rgba="0.025 0.025 0.025 1",
                      specular="0.65", shininess="0.35")
        ET.SubElement(assets, "mesh", name="cap_mesh", file=str(cap_path))
        ET.SubElement(world, "geom", name="pen_cap", type="mesh", mesh="cap_mesh",
                      material="cap_material", contype="0", conaffinity="0", mass="0")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def main():
    choices = sorted(p.name for p in (ROOT / "assets").iterdir()
                     if (p / f"{p.name}.obj").is_file() and (p / f"{p.name}.png").is_file())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, choices=choices)
    args = parser.parse_args()
    try:
        select_renderer()
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")

    import mujoco
    import mujoco.viewer

    model = load_asset(args.asset)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data, show_left_ui=False, show_right_ui=False)


if __name__ == "__main__":
    main()
