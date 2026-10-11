"""Headless sphere-tracing renderer for GLSL signed distance functions.

User code defines `float sdf(vec3 p)` (negative inside, Y up, shape
inside [-1, 1]^3, ground plane at y = -1) and may also define
`vec3 sdf_color(vec3 p)`. It can call everything in sdf_lib.glsl.
Rendering is one OpenGL 3.3 fragment shader run offscreen via moderngl:
EGL on Linux (no display needed), the platform's native GL elsewhere.

CLI (harness.py runs this in a subprocess with a timeout, so a bad
shader can't hang or crash the caller):
    python render.py shape.glsl out.png [--size 768] [--views 1|4] [--aa 2] [--no-shadows]
                     [--gif out.gif [--gif-size 384] [--gif-frames 36]]
Exit codes: 0 ok, 1 shader error (compile / nothing visible),
3 no usable OpenGL context.
"""
import argparse
import math
import os
import re
import sys

import moderngl
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "sdf_lib.glsl")) as _f:
    SDF_LIB = _f.read()

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    uv = in_pos;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_HEADER = """#version 330
in vec2 uv;
out vec4 frag_color;
uniform vec3 cam_pos;
uniform bool shadows;
"""

# `#line 10000` keeps footer errors distinguishable from user-code lines.
FRAGMENT_FOOTER = """
#line 10000
const float FLOOR_Y = -1.0;

vec3 calc_normal(vec3 p) {
    // tetrahedron technique: 4 sdf() taps instead of 6
    const vec2 k = vec2(1.0, -1.0);
    const float h = 5e-4;
    return normalize(k.xyy * sdf(p + k.xyy * h) + k.yyx * sdf(p + k.yyx * h)
                   + k.yxy * sdf(p + k.yxy * h) + k.xxx * sdf(p + k.xxx * h));
}

float soft_shadow(vec3 ro, vec3 rd) {
    float res = 1.0;
    float t = 0.02;
    for (int i = 0; i < 96; i++) {
        float h = sdf(ro + rd * t);
        res = min(res, 10.0 * h / t);
        t += clamp(h, 0.01, 0.25);
        if (res < 0.002 || t > 8.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

float ambient_occlusion(vec3 p, vec3 n) {
    float occ = 0.0;
    float w = 1.0;
    for (int i = 1; i <= 5; i++) {
        float h = 0.03 * float(i);
        occ += w * (h - min(sdf(p + n * h), p.y + n.y * h - FLOOR_Y));
        w *= 0.7;
    }
    return clamp(1.0 - 2.5 * occ, 0.0, 1.0);
}

vec3 base_color(vec3 p) {
#ifdef HAS_COLOR
    return clamp(sdf_color(p), 0.0, 1.0);
#else
    return vec3(0.85, 0.55, 0.3);
#endif
}

vec3 shade(vec3 p, vec3 n, vec3 rd, vec3 albedo) {
    vec3 l = normalize(vec3(0.5, 0.8, 0.35));
    float dif = max(dot(n, l), 0.0);
    if (shadows && dif > 0.0) dif *= soft_shadow(p + n * 2e-3, l);
    float sky = 0.5 + 0.5 * n.y;
    float occ = ambient_occlusion(p, n);
    float spe = pow(max(dot(reflect(rd, n), l), 0.0), 32.0) * dif;
    vec3 col = albedo * (1.6 * dif * vec3(1.0, 0.95, 0.85)
                         + 0.5 * sky * occ * vec3(0.6, 0.7, 0.9));
    return col + 0.2 * spe;
}

void main() {
    vec3 fwd = normalize(-cam_pos);
    vec3 right = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, fwd);
    vec3 rd = normalize(fwd * 2.0 + right * uv.x + up * uv.y);
    vec3 ro = cam_pos;
    vec3 bg = mix(vec3(0.16, 0.17, 0.2), vec3(0.32, 0.35, 0.4), 0.5 + 0.5 * uv.y);

    // 0 = miss, 1 = object, 2 = floor
    int hit = 0;
    float t = 0.0;
    for (int i = 0; i < 300; i++) {
        vec3 p = ro + rd * t;
        float d_obj = sdf(p);
        float d_floor = p.y - FLOOR_Y;
        float d = min(d_obj, d_floor);
        if (d < 1e-4 * max(t, 1.0)) { hit = d_obj < d_floor ? 1 : 2; break; }
        t += d * 0.9;  // under-relax: LLM SDFs are often not exact distances
        if (t > 20.0) break;
    }

    vec3 col = bg;
    float alpha = 0.0;  // alpha marks object pixels, for the coverage check
    vec3 p = ro + rd * t;
    if (hit == 1) {
        col = shade(p, calc_normal(p), rd, base_color(p));
        alpha = 1.0;
    } else if (hit == 2) {
        col = shade(p, vec3(0.0, 1.0, 0.0), rd, vec3(0.5));
        col = mix(col, bg, 1.0 - exp(-0.02 * t * t));
    }
    frag_color = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.4545)), alpha);
}
"""

# (azimuth, elevation) in degrees, keyed by view count
VIEWS = {1: [(35, 22)], 4: [(35, 22), (125, 22), (215, 22), (35, 70)]}
CAMERA_DISTANCE = 3.1

HAS_COLOR_RE = re.compile(r"\bvec3\s+sdf_color\s*\(")


class ShaderError(RuntimeError):
    """The shape code is at fault; worth showing to the LLM."""


class GLContextError(RuntimeError):
    """The machine can't render at all; retrying the LLM won't help."""


def build_fragment_shader(user_code: str) -> str:
    defines = "#define HAS_COLOR\n" if HAS_COLOR_RE.search(user_code) else ""
    # `#line 1` makes compile errors use the user's own line numbers
    return FRAGMENT_HEADER + defines + SDF_LIB + "\n#line 1\n" + user_code + "\n" + FRAGMENT_FOOTER


def annotate_compile_error(log: str, user_code: str) -> str:
    """Appends the offending source lines, since compilers only give numbers.
    Matches NVIDIA `0(39)` and Mesa `0:39(12)` line references."""
    lines = user_code.splitlines()
    matches = re.findall(r"\b0(?:\((\d+)\)|:(\d+)\()", log)
    numbers = sorted({int(nvidia or mesa) for nvidia, mesa in matches})
    quoted = [f"  line {n}: {lines[n - 1].strip()}" for n in numbers if 1 <= n <= len(lines)]
    return log.strip() + ("\nOffending lines:\n" + "\n".join(quoted) if quoted else "")


def create_context():
    attempts = [{"backend": "egl"}, {}] if sys.platform.startswith("linux") else [{}]
    errors = []
    for kwargs in attempts:
        try:
            return moderngl.create_standalone_context(require=330, **kwargs)
        except Exception as e:
            errors.append(f"{kwargs.get('backend', 'native')}: {e}")
    raise GLContextError("No OpenGL 3.3 context available (" + "; ".join(errors) + ")")


def camera_position(azimuth: float, elevation: float):
    az, el = math.radians(azimuth), math.radians(elevation)
    return (
        CAMERA_DISTANCE * math.cos(el) * math.sin(az),
        CAMERA_DISTANCE * math.sin(el),
        CAMERA_DISTANCE * math.cos(el) * math.cos(az),
    )


def render_views(user_code: str, size: int = 768, views=1, aa: int = 2, shadows: bool = True):
    """Returns one RGBA uint8 array per view, rendered at size*aa.
    `views` is a key of VIEWS or a list of (azimuth, elevation) pairs."""
    cameras = VIEWS[views] if isinstance(views, int) else views
    if not re.search(r"\bfloat\s+sdf\s*\(\s*vec3\s+\w+\s*\)", user_code):
        raise ShaderError("The code must define the function `float sdf(vec3 p)`.")
    ctx = create_context()
    try:
        try:
            prog = ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=build_fragment_shader(user_code),
            )
        except moderngl.Error as e:
            raise ShaderError("GLSL compile error (line numbers are lines of your code):\n"
                              + annotate_compile_error(str(e), user_code)) from None

        px = size * aa
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        vao = ctx.vertex_array(prog, [(ctx.buffer(quad.tobytes()), "2f", "in_pos")])
        fbo = ctx.simple_framebuffer((px, px), components=4)
        fbo.use()
        prog["shadows"].value = shadows

        images = []
        for az, el in cameras:
            prog["cam_pos"].value = camera_position(az, el)
            fbo.clear()
            vao.render(moderngl.TRIANGLE_STRIP)
            img = np.frombuffer(fbo.read(components=4), dtype=np.uint8)
            images.append(np.flipud(img.reshape(px, px, 4)))
        return images
    finally:
        ctx.release()


def check_coverage(images, min_frac: float = 0.003, max_frac: float = 0.95, check_border: bool = True):
    """Catches shaders that compile but draw nothing, fill the screen, or
    run off the edge of the frame (shape outside the [-1, 1]^3 box)."""
    coverage = [float((img[..., 3] > 0).mean()) for img in images]
    if max(coverage) < min_frac:
        raise ShaderError(
            "The shader compiled but the shape is not visible from any view. "
            "Make sure sdf(p) is negative inside the shape and the shape lies "
            "within the [-1, 1]^3 box."
        )
    if min(coverage) > max_frac:
        raise ShaderError(
            "The shape fills the whole view. sdf(p) is probably negative almost "
            "everywhere (wrong sign, or the shape is far larger than [-1, 1]^3)."
        )
    if check_border:
        for img in images:
            alpha = img[..., 3]
            edges = np.concatenate([alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]])
            if (edges > 0).mean() > 0.01:
                raise ShaderError(
                    "The shape is cut off by the edge of the image: it is not "
                    "centered or is larger than the [-1, 1]^3 box. Keep every part "
                    "within |x| < 0.9, |z| < 0.9 and -1 < y < 0.9, centered on the "
                    "origin."
                )
    return coverage


def _to_rgb(img, size):
    return Image.fromarray(np.ascontiguousarray(img[..., :3])).resize((size, size), Image.LANCZOS)


def render_turntable(user_code: str, output_path: str, size: int = 384, frames: int = 36,
                     aa: int = 2, shadows: bool = True, elevation: float = 22, fps: int = 18):
    """Orbits the camera 360 degrees and saves a looping GIF."""
    cameras = [(VIEWS[1][0][0] + 360.0 * i / frames, elevation) for i in range(frames)]
    images = render_views(user_code, size, cameras, aa, shadows)
    check_coverage(images, check_border=False)  # the still image already enforced framing
    gif = [_to_rgb(img, size) for img in images]
    gif[0].save(output_path, save_all=True, append_images=gif[1:],
                duration=round(1000 / fps), loop=0, optimize=True)


def render_to_png(user_code: str, output_path: str, size: int = 768, views: int = 1,
                  aa: int = 2, shadows: bool = True):
    images = render_views(user_code, size, views, aa, shadows)
    coverage = check_coverage(images)
    tiles = [_to_rgb(img, size) for img in images]
    cols = 1 if len(tiles) == 1 else 2
    rows = len(tiles) // cols
    sheet = Image.new("RGB", (cols * size, rows * size))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * size, (i // cols) * size))
    sheet.save(output_path)
    return coverage


def main():
    parser = argparse.ArgumentParser(description="Render a GLSL SDF to PNG.")
    parser.add_argument("shader", help="file defining float sdf(vec3 p)")
    parser.add_argument("output", help="output .png path")
    parser.add_argument("--size", type=int, default=768, help="pixels per view (default 768)")
    parser.add_argument("--views", type=int, choices=sorted(VIEWS), default=1)
    parser.add_argument("--aa", type=int, default=2, help="supersampling factor (default 2)")
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--gif", metavar="PATH", help="also save a turntable GIF here")
    parser.add_argument("--gif-size", type=int, default=384)
    parser.add_argument("--gif-frames", type=int, default=36)
    args = parser.parse_args()

    with open(args.shader) as f:
        code = f.read()
    try:
        render_to_png(code, args.output, args.size, args.views, args.aa, not args.no_shadows)
        if args.gif:
            render_turntable(code, args.gif, args.gif_size, args.gif_frames, args.aa, not args.no_shadows)
    except ShaderError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except GLContextError as e:
        print(e, file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
