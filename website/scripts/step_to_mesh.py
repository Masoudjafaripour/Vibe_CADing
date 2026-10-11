"""Tessellate a STEP file into a small JSON mesh (+ B-rep edges) for the
website's three.js viewer. Run once per model; the output is committed.

    pip install cadquery
    python website/scripts/step_to_mesh.py src/CAD_Harness/cadquery/plate.step \
        website/public/models/plate.json
"""
import json
import sys

import cadquery as cq


def main(step_path, out_path, tolerance=0.02, angular_tolerance=0.1, edge_samples=24):
    shape = cq.importers.importStep(step_path).val()
    verts, tris = shape.tessellate(tolerance, angular_tolerance)

    edges = []
    for edge in shape.Edges():
        n = 2 if edge.geomType() == "LINE" else edge_samples
        pts = [edge.positionAt(i / (n - 1)) for i in range(n)]
        edges.append([round(c, 4) for p in pts for c in p.toTuple()])

    bb = shape.BoundingBox()
    data = {
        "source": step_path,
        "units": "mm",
        "bbox": [bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax],
        "faces": len(shape.Faces()),
        "edges_count": len(edges),
        "positions": [round(c, 4) for v in verts for c in v.toTuple()],
        "indices": [i for t in tris for i in t],
        "edges": edges,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"{out_path}: {len(verts)} vertices, {len(tris)} triangles, "
          f"{data['faces']} faces, {len(edges)} edges")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
