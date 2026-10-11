"""Headless sphere-tracing renderer for a GLSL signed distance function.

The user code must define `float sdf(vec3 p)`, with the shape fitting
inside the [-1, 1]^3 box. Renders four views into one PNG via an EGL
OpenGL context (no display server needed).

CLI (used by harness.py so GL crashes stay in a subprocess):
    python render.py shape.glsl preview.png
Exits non-zero with the GLSL compile log / sanity-check error on stderr.
"""
import math
import sys

import moderngl
import numpy as np
from PIL import Image

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    uv = in_pos;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_HEADER = """
#version 330
in vec2 uv;
out vec4 frag_color;
uniform vec3 cam_pos;
uniform float aspect;
"""

# Line offset of user code inside the full fragment shader, so compile
# errors can be reported against the user's own line numbers.
USER_LINE_OFFSET = FRAGMENT_HEADER.count("\n")

FRAGMENT_FOOTER = """
vec3 calc_normal(vec3 p) {
    const vec2 e = vec2(1e-3, 0.0);
    return normalize(vec3(
        sdf(p + e.xyy) - sdf(p - e.xyy),
        sdf(p + e.yxy) - sdf(p - e.yxy),
        sdf(p + e.yyx) - sdf(p - e.yyx)));
}

void main() {
    vec3 fwd = normalize(-cam_pos);
    vec3 right = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, fwd);
    vec3 rd = normalize(fwd * 1.8 + right * uv.x * aspect + up * uv.y);

    float t = 0.0;
    bool hit = false;
    for (int i = 0; i < 256; i++) {
        float d = sdf(cam_pos + rd * t);
        if (d < 1e-4 * max(t, 1.0)) { hit = true; break; }
        t += d * 0.9;  // under-relax: LLM SDFs are often not exact distances
        if (t > 10.0) break;
    }

    vec3 bg = vec3(0.12, 0.13, 0.15);
    if (!hit) { frag_color = vec4(bg, 0.0); return; }

    vec3 p = cam_pos + rd * t;
    vec3 n = calc_normal(p);
    vec3 light = normalize(vec3(0.6, 0.8, 0.4));
    float diff = max(dot(n, light), 0.0);
    float rim = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
    vec3 col = vec3(0.85, 0.55, 0.3) * (0.2 + 0.8 * diff) + 0.15 * rim;
    frag_color = vec4(pow(col, vec3(0.4545)), 1.0);
}
"""

# (azimuth, elevation) in degrees for the 2x2 preview grid
VIEWS = [(30, 20), (120, 20), (210, 20), (30, 70)]


class SDFRenderError(RuntimeError):
    pass


def build_fragment_shader(user_code: str) -> str:
    return FRAGMENT_HEADER + user_code + "\n" + FRAGMENT_FOOTER


def render_views(user_code: str, size: int = 384, distance: float = 3.2):
    """Returns a list of RGBA uint8 arrays, one per entry in VIEWS."""
    ctx = moderngl.create_standalone_context(backend="egl")
    try:
        try:
            prog = ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=build_fragment_shader(user_code),
            )
        except moderngl.Error as e:
            raise SDFRenderError(
                f"GLSL compile error (user code starts after line "
                f"{USER_LINE_OFFSET} of the full shader):\n{e}"
            ) from None

        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        vao = ctx.vertex_array(prog, [(ctx.buffer(quad.tobytes()), "2f", "in_pos")])
        fbo = ctx.simple_framebuffer((size, size), components=4)
        fbo.use()
        prog["aspect"].value = 1.0

        images = []
        for az, el in VIEWS:
            az, el = math.radians(az), math.radians(el)
            prog["cam_pos"].value = (
                distance * math.cos(el) * math.sin(az),
                distance * math.sin(el),
                distance * math.cos(el) * math.cos(az),
            )
            fbo.clear()
            vao.render(moderngl.TRIANGLE_STRIP)
            img = np.frombuffer(fbo.read(components=4), dtype=np.uint8)
            images.append(np.flipud(img.reshape(size, size, 4)))
        return images
    finally:
        ctx.release()


def check_coverage(images, min_frac: float = 0.005, max_frac: float = 0.95):
    """Catches shaders that compile but draw nothing or fill the screen."""
    coverage = [float((img[..., 3] > 0).mean()) for img in images]
    if max(coverage) < min_frac:
        raise SDFRenderError(
            "The shader compiled but the surface is not visible from any "
            "view. Make sure sdf(p) is negative inside the shape and the "
            "shape lies within the [-1, 1]^3 box."
        )
    if min(coverage) > max_frac:
        raise SDFRenderError(
            "The surface fills the whole view. sdf(p) is probably negative "
            "almost everywhere (wrong sign, or the shape is far larger "
            "than the [-1, 1]^3 box)."
        )
    return coverage


def render_preview(user_code: str, output_path: str, size: int = 384):
    images = render_views(user_code, size)
    coverage = check_coverage(images)
    top = np.concatenate(images[:2], axis=1)
    bottom = np.concatenate(images[2:], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    Image.fromarray(grid[..., :3]).save(output_path)
    return coverage


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    with open(sys.argv[1]) as f:
        code = f.read()
    try:
        render_preview(code, sys.argv[2])
    except SDFRenderError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
