"""GLSL SDF -> watertight, 3D-printable STL.

1. Samples the shape's sdf() on a 3D grid on the GPU (one fragment-shader pass
   per z-slice, same sdf_lib.glsl as the renderer).
2. Meshes the grid with the C++ geometry engine (vibecad_geom: marching cubes,
   watertight / genus checks, binary STL).

    python to_stl.py shape.glsl shape.stl [--res 128] [--scale 50]
    python to_stl.py shape.glsl shape.stl --simplify 20000 --glb shape.glb   # lighter mesh + coloured glTF

By default the mesh is rotated to Z-up, scaled so 1 SDF unit = 50 mm (shapes
come out ~10 cm tall) and placed on the build plate (z = 0), ready for a slicer.
Exit codes: 0 ok, 1 shader error, 3 no OpenGL, 4 vibecad_geom not installed.
"""
import argparse
import json
import os
import re
import sys
import time

import moderngl
import numpy as np

from render import (HAS_COLOR_RE, SDF_LIB, VERTEX_SHADER, GLContextError, ShaderError, annotate_compile_error,
                    create_context)

SAMPLE_HEADER = """#version 330
in vec2 uv;
out float value;
uniform float z;
uniform float lo;
uniform float h;
"""

# Pixel (i, j) of slice k holds sdf(lo + i*h, lo + j*h, lo + k*h).
SAMPLE_FOOTER = """
#line 10000
void main() {
    vec2 ij = floor(gl_FragCoord.xy);
    value = sdf(vec3(lo + ij.x * h, lo + ij.y * h, z));
}
"""


def sample_sdf(user_code: str, res: int, lo: float, hi: float) -> np.ndarray:
    """Returns sdf values on an (res+1)^3 grid over [lo, hi]^3, indexed [z, y, x]."""
    if not re.search(r"\bfloat\s+sdf\s*\(\s*vec3\s+\w+\s*\)", user_code):
        raise ShaderError("The code must define the function `float sdf(vec3 p)`.")
    m = res + 1
    h = (hi - lo) / res
    ctx = create_context()
    try:
        try:
            prog = ctx.program(vertex_shader=VERTEX_SHADER,
                               fragment_shader=SAMPLE_HEADER + SDF_LIB + "\n#line 1\n" + user_code + "\n" + SAMPLE_FOOTER)
        except moderngl.Error as e:
            raise ShaderError("GLSL compile error (line numbers are lines of your code):\n"
                              + annotate_compile_error(str(e), user_code)) from None
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        vao = ctx.vertex_array(prog, [(ctx.buffer(quad.tobytes()), "2f", "in_pos")])
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((m, m), 1, dtype="f4")])
        fbo.use()
        prog["lo"].value = lo
        prog["h"].value = h

        grid = np.empty((m, m, m), dtype=np.float32)
        for k in range(m):
            prog["z"].value = lo + k * h
            vao.render(moderngl.TRIANGLE_STRIP)
            grid[k] = np.frombuffer(fbo.read(components=1, dtype="f4"), dtype=np.float32).reshape(m, m)
        return grid
    finally:
        ctx.release()


COLOR_FOOTER = """
#line 10000
uniform sampler2D points;
void main() {
    vec3 p = texelFetch(points, ivec2(gl_FragCoord.xy), 0).xyz;
#ifdef HAS_COLOR
    value = vec4(clamp(sdf_color(p), 0.0, 1.0), 1.0);
#else
    value = vec4(0.85, 0.55, 0.3, 1.0);  // render.py's default albedo
#endif
}
"""


def sample_colors(user_code: str, points: np.ndarray, width: int = 1024) -> np.ndarray:
    """Evaluates the shape's sdf_color(p) at each point (N, 3) on the GPU -> (N, 3) RGB in 0-1."""
    n = len(points)
    height = -(-n // width)
    packed = np.zeros((height * width, 4), dtype=np.float32)
    packed[:n, :3] = points
    defines = "#define HAS_COLOR\n" if HAS_COLOR_RE.search(user_code) else ""
    header = SAMPLE_HEADER.replace("out float value;", "out vec4 value;")
    ctx = create_context()
    try:
        prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=header + defines + SDF_LIB + "\n#line 1\n"
                           + user_code + "\n" + COLOR_FOOTER)
        tex = ctx.texture((width, height), 4, packed.tobytes(), dtype="f4")
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.use(0)
        if "points" in prog:  # optimised out when sdf_color() ignores p (a constant colour)
            prog["points"].value = 0
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        vao = ctx.vertex_array(prog, [(ctx.buffer(quad.tobytes()), "2f", "in_pos")])
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((width, height), 4, dtype="f4")])
        fbo.use()
        vao.render(moderngl.TRIANGLE_STRIP)
        rgba = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32).reshape(-1, 4)
        return rgba[:n, :3].copy()
    finally:
        ctx.release()


def close_boundary(grid: np.ndarray) -> int:
    """Marks the outer grid layer as outside so the mesh is always closed.
    Returns how many boundary samples were inside (shape clipped by the box)."""
    np.nan_to_num(grid, copy=False, nan=1.0, posinf=1.0, neginf=-1.0)
    clipped = 0
    for face in (np.s_[0], np.s_[-1], np.s_[:, 0], np.s_[:, -1], np.s_[:, :, 0], np.s_[:, :, -1]):
        clipped += int((grid[face] <= 0).sum())
        grid[face] = np.maximum(grid[face], 1e-4)
    return clipped


def to_print_frame(vertices: np.ndarray, scale: float) -> np.ndarray:
    """Y-up SDF space -> Z-up millimetres resting on z = 0 (a pure rotation, so winding is kept)."""
    v = np.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], axis=1) * scale
    v[:, 2] -= v[:, 2].min()
    return v.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert a GLSL SDF shape to a watertight STL mesh.")
    parser.add_argument("shader", help="file defining float sdf(vec3 p)")
    parser.add_argument("output", help="output .stl path")
    parser.add_argument("--res", type=int, default=128, help="grid cells per axis, 16-512 (default 128)")
    parser.add_argument("--margin", type=float, default=0.1, help="padding around the [-1, 1]^3 box (default 0.1)")
    parser.add_argument("--scale", type=float, default=50.0, help="mm per SDF unit (default 50)")
    parser.add_argument("--y-up", action="store_true", help="keep the SDF's Y-up axes and units (no print transform)")
    parser.add_argument("--json", metavar="PATH", help="also write the mesh stats as JSON")
    parser.add_argument("--simplify", type=int, metavar="TRIANGLES",
                        help="reduce to about this many triangles (QEM; keeps genus and shells)")
    parser.add_argument("--smooth", type=int, default=0, metavar="ITERATIONS",
                        help="Taubin smoothing iterations (for noisy shapes; marching cubes rarely needs it)")
    parser.add_argument("--glb", metavar="PATH", help="also write a coloured glTF (.glb): Y up, metres, on the ground")
    args = parser.parse_args()
    if not 16 <= args.res <= 512:
        parser.error("--res must be between 16 and 512")

    try:
        import vibecad_geom
    except ImportError:
        print("vibecad_geom is not installed. From the repo root run: pip install ./src/geometry_engine",
              file=sys.stderr)
        sys.exit(4)

    with open(args.shader) as f:
        code = f.read()
    lo, hi = -1.0 - args.margin, 1.0 + args.margin
    try:
        t0 = time.perf_counter()
        grid = sample_sdf(code, args.res, lo, hi)
        t_sample = time.perf_counter() - t0
    except ShaderError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except GLContextError as e:
        print(e, file=sys.stderr)
        sys.exit(3)

    clipped = close_boundary(grid)
    t0 = time.perf_counter()
    vertices, triangles = vibecad_geom.marching_cubes(grid, lo, hi)
    t_mesh = time.perf_counter() - t0
    if len(triangles) == 0:
        print("The shape has no surface inside the sampled box (sdf() is never negative).", file=sys.stderr)
        sys.exit(1)

    full_triangles = len(triangles)
    if args.smooth:
        vertices = vibecad_geom.smooth_taubin(vertices, triangles, args.smooth)
    if args.simplify and args.simplify < len(triangles):
        vertices, triangles, _ = vibecad_geom.simplify(vertices, triangles, args.simplify)
    if args.glb:  # glTF convention: Y up, metres; resting on the ground
        rgb = sample_colors(code, vertices)
        meters = (vertices * args.scale / 1000).astype(np.float32)
        meters[:, 1] -= meters[:, 1].min()
        vibecad_geom.write_glb(meters, triangles, args.glb, rgb, os.path.splitext(os.path.basename(args.glb))[0])

    unit = "units"
    if not args.y_up:
        vertices, unit = to_print_frame(vertices, args.scale), "mm"
    stats = vibecad_geom.analyze(vertices, triangles)
    vibecad_geom.write_stl(vertices, triangles, args.output)

    size = vertices.max(axis=0) - vertices.min(axis=0)
    simplified = f" (simplified from {full_triangles:,})" if len(triangles) < full_triangles else ""
    print(f"{args.output}: {len(vertices):,} vertices, {len(triangles):,} triangles{simplified} "
          f"({args.res}^3 grid; GPU sampling {t_sample * 1000:.0f} ms, C++ meshing {t_mesh * 1000:.0f} ms)")
    print(f"  size {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} {unit} | volume {stats['volume']:.1f} {unit}^3 | "
          f"watertight {'yes' if stats['watertight'] else 'no'} | shells {stats['components']} | genus {stats['genus']}")
    if clipped:
        print(f"  note: the shape extends past the sampled box; {clipped} boundary samples were capped")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({**stats, "vertices": len(vertices), "triangles": len(triangles), "res": args.res,
                       "size": [round(float(x), 2) for x in size], "unit": unit, "clipped": clipped,
                       "sample_ms": round(t_sample * 1000, 1), "mesh_ms": round(t_mesh * 1000, 1)}, f, indent=2)


if __name__ == "__main__":
    main()
