# Shortcuts for the bookshelf scene. Run `just` to list them.

# Python: the repo's own .venv, else ../.venv (the original workspace), else python3
py := if path_exists(".venv/bin/python") == "true" { ".venv/bin/python" } else if path_exists("../.venv/bin/python") == "true" { "../.venv/bin/python" } else { "python3" }
# live viewers launched from a script need mjpython on macOS
mjpy := if os() == "macos" { replace(py, "bin/python", "bin/mjpython") } else { py }

# list the shortcuts
default:
    @just --list --unsorted

# open the viewer in a pose: home (default) or ready
view pose="home":
    {{py}} view.py {{pose}}

# open the viewer in the ready pose
ready:
    {{py}} view.py ready

# watch a scripted grasp live: table (cup) or book
watch test="table":
    {{mjpy}} tests/grasp_test.py {{test}} --view

# run grasp tests headless and write GIFs to tests/: table, book, or both (default)
grasp *tests="table book":
    {{py}} tests/grasp_test.py {{tests}}

# rebuild meshes, robot and scene.xml from inputs/ and scan_measurements.json
build:
    {{py}} build_scene.py

# measure a capture zip (or mesh) and save scan_measurements.json
measure capture:
    {{py}} tools/measure_scan.py {{capture}} --write

# new capture end to end: measure it, then rebuild the scene
rebuild capture: (measure capture) build

# regenerate the LEAP hand model from its URDF (only if the URDF changes)
hand:
    {{py}} tools/convert_leap.py

# first-time setup on a new machine: create .venv and install requirements
setup:
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
