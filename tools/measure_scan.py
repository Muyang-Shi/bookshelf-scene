"""Measure table / shelf / UR5 mount from a metric scene scan.

Accepts a phone-scanner export zip (e.g. 3D Scanner App: textured_output.obj + export_refined.obj),
or individual meshes (textured OBJ, PLY with vertex colors). With a zip, both meshes are measured and
averaged. Numbers are in the scene frame build_scene.py uses: origin on the floor under the table
center, +Y toward the shelf, +Z up (right-handed).

Usage:
  python tools/measure_scan.py <capture.zip | mesh.obj | mesh.ply> [more meshes...] [--write]

  --write   save the averaged result to scan_measurements.json next to build_scene.py;
            build_scene.py picks it up automatically on the next build.
"""
import json
import os
import sys
import tempfile
import zipfile
import numpy as np
import trimesh
from scipy import ndimage

rng = np.random.default_rng(0)

# Height of the UR5 shoulder cap's lowest point above the robot's mounting face, taken from the
# MuJoCo model. The cap is the lowest blue surface on the arm, whatever the joint angles.
SHOULDER_CAP_ABOVE_MOUNT = 0.024
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scan_measurements.json")


def load_points(path, n=1_500_000):
    """Area-uniform surface samples with colors. Raw vertices are not enough: scanner apps
    decimate flat regions (e.g. a table top) down to a few large triangles."""
    m = trimesh.load(path, force="mesh", process=False)
    if len(m.faces) == 0:
        return np.asarray(m.vertices, float), np.asarray(m.visual.vertex_colors)[:, :3].astype(float)
    if m.visual.kind == "texture":
        V, _, C = trimesh.sample.sample_surface(m, n, sample_color=True, seed=0)
    else:
        V, fi = trimesh.sample.sample_surface(m, n, seed=0)
        if m.visual.kind == "vertex":
            C = np.asarray(m.visual.vertex_colors)[m.faces[fi]][:, :, :3].mean(1)   # face-average color
        else:
            C = np.full((len(V), 3), 128.0)
    return np.asarray(V, float), np.asarray(C)[:, :3].astype(float)


def ransac_plane(P, iters=3000, thresh=0.01, up_hint=None):
    best, bn = None, 0
    sample = P[rng.choice(len(P), min(len(P), 200000), replace=False)]
    for _ in range(iters):
        s = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        l = np.linalg.norm(n)
        if l < 1e-9:
            continue
        n /= l
        if up_hint is not None and abs(n @ up_hint) < 0.95:   # floor must be ~horizontal
            continue
        d = -n @ s[0]
        c = (np.abs(sample @ n + d) < thresh).sum()
        if c > bn:
            best, bn = (n, d), c
    n, d = best
    # the floor is the lowest large plane: orient the normal so most points are above it
    if np.median(P @ n + d) < 0:
        n, d = -n, -d
    return n, d


def min_area_rect(xy):
    best = None
    for a in np.radians(np.arange(0, 90, 0.5)):
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        q = xy @ R
        lo, hi = np.percentile(q, 1, 0), np.percentile(q, 99, 0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = (area, R, lo, hi)
    return best[1:]


def measure(path):
    """Measure one mesh; returns a dict of scene numbers (plus raw arrays under 'W', 'C', 'h')."""
    V, C = load_points(path)
    print(f"{path}\n  {len(V)} surface samples, extents {np.round(np.ptp(V, 0), 2)}")

    # 1. floor plane (scans are Y-up: ARKit / Record3D)
    n, d = ransac_plane(V, up_hint=np.array([0, 1.0, 0]))
    h = V @ n + d
    print(f"  floor normal {np.round(n, 3)}, {(np.abs(h) < 0.01).mean() * 100:.0f}% of samples on it")

    # 2. table top = strongest horizontal slab between 0.5 and 1.0 m
    hist, edges = np.histogram(h[(h > 0.5) & (h < 1.0)], bins=np.arange(0.5, 1.0, 0.005))
    top_h = edges[np.argmax(hist)] + 0.0025
    print(f"  table top height {top_h:.3f} m")

    # horizontal basis
    a = np.array([1.0, 0, 0]) - n * n[0]
    ex = a / np.linalg.norm(a)
    ey = np.cross(n, ex)
    P2 = np.c_[V @ ex, V @ ey]

    slab = np.abs(h - top_h) < 0.012
    g = 0.02
    xy = P2[slab]
    mn = xy.min(0)
    ij = ((xy - mn) / g).astype(int)
    G = np.zeros(ij.max(0) + 1, bool)
    G[ij[:, 0], ij[:, 1]] = True
    G = ndimage.binary_closing(G, iterations=2)
    L, _ = ndimage.label(G)
    sizes = np.bincount(L.ravel())
    sizes[0] = 0
    keep = L[ij[:, 0], ij[:, 1]] == np.argmax(sizes)
    R, lo, hi = min_area_rect(xy[keep])
    c = (lo + hi) / 2
    size = hi - lo
    Q = (P2 - (c @ R.T)) @ R          # table-aligned 2D coords, origin at table center

    # 3. shelf: stuff standing on the table
    inside = (np.abs(Q[:, 0]) < size[0] / 2) & (np.abs(Q[:, 1]) < size[1] / 2)
    above = inside & (h > top_h + 0.03) & (h < top_h + 1.2)
    S = Q[above]
    sh = h[above]
    top = np.percentile(sh, 99.5)
    off = np.median(S, 0) / (size / 2)
    depth_axis = int(np.argmax(np.abs(off)))          # shelf sits against one edge
    sgn = np.sign(off[depth_axis])
    # scene frame: +Y toward the shelf, X = Y x Z (right-handed)
    Yv = np.zeros(2); Yv[depth_axis] = sgn
    Xv = np.array([Yv[1], -Yv[0]])
    W = np.c_[Q @ Xv, Q @ Yv, h]
    table_w, table_d = size[1 - depth_axis], size[depth_axis]
    print(f"  table top {table_w:.3f} (X, along shelf) x {table_d:.3f} (Y) m")

    band = W[above & (h > top - 0.08) & (h < top)]
    sx = np.percentile(band[:, 0], [2, 98])
    sy = np.percentile(band[:, 1], [2, 98])
    print(f"  shelf top at {top:.3f} m -> height {top - top_h:.3f} m")
    print(f"  shelf width {sx[1] - sx[0]:.3f} m, center X {sx.mean():+.3f}, "
          f"depth (top band) {sy[1] - sy[0]:.3f} m, back at Y {sy[1]:+.3f} (table edge {table_d / 2:+.3f})")

    # 4. UR5: blue joint caps outside the table, near it
    blue = (C[:, 2] > C[:, 0] + 25) & (C[:, 2] > 110)
    out = ~inside & (np.hypot(W[:, 0], W[:, 1]) < 1.6) & (h > 0.3) & (h < 1.6)
    B = W[blue & out]
    res = dict(table_w=float(table_w), table_d=float(table_d), table_h=float(top_h),
               shelf_w=float(sx[1] - sx[0]), shelf_d=float(sy[1] - sy[0]), shelf_h=float(top - top_h),
               shelf_x=float(sx.mean()), shelf_back_gap=float(table_d / 2 - sy[1]), W=W, C=C, h=h)
    if len(B) < 20:
        print("  UR5: too few blue points found (mesh has no color?)")
        return res
    grid = 0.1
    k = np.floor(B[:, :2] / grid).astype(int)
    keys, cnt = np.unique(k, axis=0, return_counts=True)
    cell = keys[np.argmax(cnt)]
    near = B[np.all(np.abs(np.floor(B[:, :2] / grid) - cell) <= 2, axis=1)]
    zlow = np.percentile(near[:, 2], 3)
    low = near[near[:, 2] < zlow + 0.15]
    print(f"  UR5 blue caps: {len(near)} pts, lowest cap band {zlow:.3f}-{zlow + 0.15:.3f} m "
          f"at XY {np.round(low[:, :2].mean(0), 3)}")
    # pedestal column under the lowest cap
    col = W[out & (np.hypot(W[:, 0] - low[:, 0].mean(), W[:, 1] - low[:, 1].mean()) < 0.2)
            & (h > 0.15) & (h < zlow - 0.02)]
    xy = np.median(col[:, :2], 0) if len(col) > 20 else low[:, :2].mean(0)
    if len(col) > 20:
        print(f"  pedestal column: {len(col)} pts, XY center {np.round(xy, 3)}")
    res.update(mount_x=float(xy[0]), mount_y=float(xy[1]), mount_z=float(zlow - SHOULDER_CAP_ABOVE_MOUNT))
    print(f"  UR5 mount estimate: ({xy[0]:+.3f}, {xy[1]:+.3f}, {zlow - SHOULDER_CAP_ABOVE_MOUNT:.3f})")
    return res


def meshes_from_zip(zpath, tmp):
    """Extract the scanner meshes (with their texture files) from a capture zip."""
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        picks = [n for n in names if os.path.basename(n).startswith("textured_output.")
                 or os.path.basename(n) == "export_refined.obj"]
        if not any(n.endswith(".obj") for n in picks):
            raise SystemExit(f"{zpath}: no textured_output.obj / export_refined.obj inside")
        for n in picks:
            out = os.path.join(tmp, os.path.basename(n))
            with z.open(n) as src, open(out, "wb") as dst:
                dst.write(src.read())
    order = ["textured_output.obj", "export_refined.obj"]
    return [os.path.join(tmp, f) for f in order if os.path.exists(os.path.join(tmp, f))]


def main(args):
    write = "--write" in args
    inputs = [a for a in args if not a.startswith("--")]
    if not inputs:
        raise SystemExit(__doc__)
    with tempfile.TemporaryDirectory() as tmp:
        meshes = []
        for p in inputs:
            meshes += meshes_from_zip(p, tmp) if p.lower().endswith(".zip") else [p]
        results = [measure(m) for m in meshes]
    avg = {}
    for k in sorted({k for r in results for k in r if k not in ("W", "C", "h")}):
        vals = [r[k] for r in results if k in r]
        avg[k] = round(float(np.mean(vals)), 4)
        if len(vals) > 1:
            print(f"  {k:15s} " + "  ".join(f"{v:.3f}" for v in vals) + f"   -> {avg[k]:.3f}")
    avg["sources"] = [os.path.basename(p) for p in inputs]
    print("\nscene numbers:", json.dumps(avg, indent=1))
    if write:
        with open(OUT_JSON, "w") as f:
            json.dump(avg, f, indent=1)
        print(f"wrote {os.path.normpath(OUT_JSON)} -- run build_scene.py to apply")


if __name__ == "__main__":
    main(sys.argv[1:])
