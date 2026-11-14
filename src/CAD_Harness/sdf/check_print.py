"""Is this shape 3D-printable? GLSL SDF -> C++ printability checks -> report.

    python check_print.py gallery/mug.glsl                 # human-readable report
    python check_print.py shape.glsl --json report.json
    python check_print.py shape.glsl --feedback            # failures as instructions for the LLM

Checks (C++, vibecad_geom): wall thickness (exact distance transform), parts that
float above the build plate, overhangs that need support, bed contact, watertight.
Units follow to_stl.py: 1 SDF unit = 50 mm, Z up, resting on the plate.
Exit codes: 0 printable (maybe with supports), 2 not printable, 1 shader error,
3 no OpenGL, 4 vibecad_geom not installed.
"""
import argparse
import json
import sys

import numpy as np

from render import GLContextError, ShaderError
from to_stl import close_boundary, sample_sdf

LO, HI = -1.1, 1.1  # sampled box in SDF units (shapes live in [-1, 1]^3)


def analyze_shape(code, res=192, scale=50.0, min_wall_mm=1.0, overhang_deg=45.0):
    """Returns the full printability report for GLSL code (raises ShaderError)."""
    import vibecad_geom as g

    grid = sample_sdf(code, res, LO, HI)
    clipped = close_boundary(grid)
    vertices, triangles = g.marching_cubes(grid, LO, HI)
    if len(triangles) == 0:
        raise ShaderError("The shape has no surface inside the box (sdf() is never negative).")

    # SDF (Y up) -> print frame (Z up, mm, on the plate); keep the shift to map locations back
    lift = float(vertices[:, 1].min() * scale)
    printed = np.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], 1) * scale
    printed[:, 2] -= lift
    report = g.check_print(printed.astype(np.float32), triangles, overhang_deg=overhang_deg)
    thickness = g.wall_thickness(grid, LO, HI, min_wall=min_wall_mm / scale)

    to_sdf = lambda p: (p[0] / scale, (p[2] + lift) / scale, -p[1] / scale)  # print mm -> SDF coords
    for issue in report["issues"]:
        issue["where_sdf"] = to_sdf(issue["where"])

    # Thin material by volume (C++ morphological opening). Under 0.1% is grid noise at
    # sharp corners; up to 2% is typically tips and edges; more is a real defect.
    t_mm = {k: thickness[k] * scale for k in ("min", "p05", "median")}
    thin = thickness["thin_volume_fraction"]
    if thin > 0.001:
        report["issues"].append({
            "kind": "thin_wall", "severity": "critical" if thin > 0.02 else "warning", "value": thin,
            "message": (f"{100 * thin:.1f}% of the material is thinner than {min_wall_mm:.1f} mm; "
                        "it may break or not print."),
            "where_sdf": tuple(thickness["thin_centroid"]), "where": None,
        })
    if clipped:
        report["issues"].append({
            "kind": "clipped", "severity": "warning", "value": clipped, "where": None,
            "message": "Part of the shape extends outside the [-1, 1]^3 box and was cut off.",
        })

    severities = {i["severity"] for i in report["issues"]}
    kinds = {i["kind"] for i in report["issues"]}
    report["verdict"] = ("not printable" if "critical" in severities
                         else "printable with supports" if "overhang" in kinds else "printable")
    report["wall_mm"] = {**t_mm, "thin_fraction": thin, "min_wall": min_wall_mm}
    report["settings"] = {"res": res, "scale_mm_per_unit": scale, "overhang_deg": overhang_deg}
    return report


def feedback(report, scale=50.0):
    """Critical issues as concrete edit instructions, in the shape's own SDF coordinates."""
    lines = []
    for i in report["issues"]:
        if i["severity"] != "critical":
            continue
        x, y, z = i.get("where_sdf") or (0, 0, 0)
        at = f"near (x={x:.2f}, y={y:.2f}, z={z:.2f})"
        if i["kind"] == "floating":
            lines.append(f"A separate part floats above the ground {at}: it would print in mid-air. "
                         "Connect it to the main body (overlap or smooth-union) or lower it to touch y = -1.")
        elif i["kind"] == "thin_wall":
            need = report["wall_mm"]["min_wall"] / scale
            lines.append(f"Walls are too thin to print {at} ({i['message']}). "
                         f"Make every part at least {need:.3f} units thick.")
        else:
            lines.append(i["message"])
    return "The shape renders, but it is not 3D-printable:\n- " + "\n- ".join(lines) if lines else ""


def print_report(report):
    w = report["wall_mm"]
    sx, sy, sz = report["size_mm"]
    print(f"verdict: {report['verdict'].upper()}")
    print(f"  size {sx:.0f} x {sy:.0f} x {sz:.0f} mm | volume {report['volume_mm3'] / 1000:.1f} cm^3 | "
          f"watertight {'yes' if report['watertight'] else 'no'} | parts {report['shells']} "
          f"({report['floating_shells']} floating)")
    print(f"  walls: thinnest {w['min']:.1f} mm, 5th percentile {w['p05']:.1f} mm, median {w['median']:.1f} mm")
    print(f"  overhangs > {report['settings']['overhang_deg']:.0f} deg: {report['overhang_area_mm2']:.0f} mm^2 "
          f"({100 * report['overhang_fraction']:.0f}% of surface) | bed contact {report['contact_area_mm2']:.0f} mm^2")
    for i in report["issues"]:
        print(f"  [{i['severity']}] {i['message']}")


def main():
    parser = argparse.ArgumentParser(description="3D-printability report for a GLSL SDF shape.")
    parser.add_argument("shader")
    parser.add_argument("--res", type=int, default=192, help="grid cells per axis (default 192: ~0.6 mm)")
    parser.add_argument("--scale", type=float, default=50.0, help="mm per SDF unit (default 50)")
    parser.add_argument("--min-wall", type=float, default=1.0, help="minimum wall thickness, mm (default 1.0)")
    parser.add_argument("--overhang", type=float, default=45.0, help="max overhang from vertical, degrees")
    parser.add_argument("--json", metavar="PATH", help="write the full report as JSON")
    parser.add_argument("--feedback", action="store_true", help="print LLM repair instructions (stderr) if not printable")
    args = parser.parse_args()

    try:
        import vibecad_geom  # noqa: F401
    except ImportError:
        sys.exit(print("vibecad_geom is not installed: pip install ./src/geometry_engine", file=sys.stderr) or 4)
    with open(args.shader) as f:
        code = f.read()
    try:
        report = analyze_shape(code, args.res, args.scale, args.min_wall, args.overhang)
    except ShaderError as e:
        sys.exit(print(e, file=sys.stderr) or 1)
    except GLContextError as e:
        sys.exit(print(e, file=sys.stderr) or 3)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    if args.feedback:
        text = feedback(report, args.scale)
        if text:
            print(text, file=sys.stderr)
    else:
        print_report(report)
    sys.exit(2 if report["verdict"] == "not printable" else 0)


if __name__ == "__main__":
    main()
