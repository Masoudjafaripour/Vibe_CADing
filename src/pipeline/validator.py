# Validation logic
"""3D-printability validation of any STL file (C++ checks in vibecad_geom).

    pip install ./src/geometry_engine            # once, from the repo root
    python src/pipeline/validator.py part.stl    # Z-up, millimetres (slicer convention)
    python src/pipeline/validator.py part.stl --y-up --scale 50

Checks: watertight, parts floating above the build plate, overhangs that need
support, bed contact, build volume. Wall thickness needs the SDF, so it is only
in src/CAD_Harness/sdf/check_print.py. Exit code 2 = not printable.
"""
import argparse
import sys

import numpy as np


def validate(path, y_up=False, scale=1.0, overhang_deg=45.0):
    import vibecad_geom as g

    vertices, triangles = g.read_stl(path)
    if y_up:  # rotate Y-up to the slicer's Z-up
        vertices = np.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], 1)
    vertices = (vertices * scale).astype(np.float32)
    vertices[:, 2] -= vertices[:, 2].min()  # rest on the build plate
    report = g.check_print(vertices, triangles, overhang_deg=overhang_deg)
    severities = {i["severity"] for i in report["issues"]}
    report["verdict"] = ("not printable" if "critical" in severities else
                         "printable with supports" if any(i["kind"] == "overhang" for i in report["issues"])
                         else "printable")
    report["triangles"] = len(triangles)
    return report


def main():
    parser = argparse.ArgumentParser(description="3D-printability check for an STL file.")
    parser.add_argument("stl")
    parser.add_argument("--y-up", action="store_true", help="the file is Y-up (rotate to Z-up)")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply coordinates to get mm")
    parser.add_argument("--overhang", type=float, default=45.0, help="max overhang from vertical, degrees")
    args = parser.parse_args()
    try:
        r = validate(args.stl, args.y_up, args.scale, args.overhang)
    except ImportError:
        sys.exit("vibecad_geom is not installed: pip install ./src/geometry_engine")
    sx, sy, sz = r["size_mm"]
    print(f"{args.stl}: {r['verdict'].upper()}")
    print(f"  {r['triangles']:,} triangles | {sx:.1f} x {sy:.1f} x {sz:.1f} mm | volume {r['volume_mm3'] / 1000:.2f} cm^3 | "
          f"watertight {'yes' if r['watertight'] else 'no'} | parts {r['shells']} ({r['floating_shells']} floating)")
    print(f"  overhangs {r['overhang_area_mm2']:.0f} mm^2 ({100 * r['overhang_fraction']:.0f}%) | "
          f"bed contact {r['contact_area_mm2']:.0f} mm^2")
    for i in r["issues"]:
        print(f"  [{i['severity']}] {i['message']}")
    sys.exit(2 if r["verdict"] == "not printable" else 0)


if __name__ == "__main__":
    main()
