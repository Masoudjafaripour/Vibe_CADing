"""Game-style renderer for saved motion clips: a stylized SDF character, GPU sphere-traced.

Same idea as the SDF harness (../CAD_Harness/sdf): the body is a signed
distance function, here built from tapered capsules between the FK joint
positions of each frame, smooth-blended per material (skin, shirt,
pants, shoes, hair). One OpenGL 3.3 fragment shader, run headless via
moderngl (EGL on Linux). The motion data is not changed.

    python render_character.py outputs/walk_ar.npz        # -> outputs/game/walk_ar.webp
    python render_character.py outputs/*.npz             # every clip
    python render_character.py --compare                  # -> outputs/game/<command>_compare.webp
    python render_character.py --grid a.npz b.npz --out name   # any clips side by side

Output is animated WebP (~0.3 MB per clip); --gif writes GIF instead (~5 MB:
the shaded gradients and moving camera compress badly into 256 colours).
"""
import argparse
import glob
import os
import sys

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from skeleton import NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs", "game")

# shirt colour per source, so procedural / AR / DDPM are easy to tell apart
SHIRT = {"procedural": (0.5, 0.5, 0.54), "ar": (0.08, 0.36, 0.85), "diffusion": (0.1, 0.6, 0.28),
         "untrained": (0.8, 0.15, 0.15), "ctrl": (0.1, 0.6, 0.28), "naive": (0.9, 0.5, 0.1),
         "footlock": (0.1, 0.55, 0.6)}
LABELS = {"procedural": "Procedural (hand-written)", "ar": "AR Transformer", "diffusion": "DDPM",
          "untrained": "Untrained network", "ctrl": "Controlled DDPM", "naive": "Naive concat",
          "footlock": "+ foot lock (post-process)"}

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() { uv = in_pos; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FRAGMENT_SHADER = """
#version 330
in vec2 uv;
out vec4 frag_color;
uniform vec3 J[%(num_joints)d];
uniform vec3 cam_pos;
uniform vec3 cam_target;
uniform vec3 facing;
uniform vec3 shirt_color;
%(joint_defines)s

float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// approximate distance to an axis-aligned ellipsoid with radii r (Inigo Quilez)
float sdEllipsoid(vec3 p, vec3 r) {
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

// capsule whose radius goes from r1 at a to r2 at b (Inigo Quilez)
float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2) {
    vec3 ba = b - a;
    float l2 = dot(ba, ba);
    float rr = r1 - r2;
    float a2 = l2 - rr * rr;
    float il2 = 1.0 / l2;
    vec3 pa = p - a;
    float y = dot(pa, ba);
    float z = y - l2;
    vec3 xv = pa * l2 - ba * y;
    float x2 = dot(xv, xv);
    float y2 = y * y * l2;
    float z2 = z * z * l2;
    float k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k) return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

// materials: 1 skin, 2 shirt, 3 pants, 4 shoes, 5 hair, 6 eyes
void add(inout vec2 res, float d, float mat, float k) {
    float m = d < res.x ? mat : res.y;
    res = vec2(smin(res.x, d, k), m);
}

vec3 head_center() { return J[HEAD] + vec3(0.0, 0.05, 0.0); }

vec2 map(vec3 p) {
    float bound = length(p - J[PELVIS] - vec3(0.0, 0.1, 0.0)) - 1.2;
    if (bound > 0.2) return vec2(bound, 0.0);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 side = normalize(cross(up, facing));
    vec3 hc = head_center();

    // skin: head, neck, forearms, hands
    float skin = length((p - hc) * vec3(1.0, 0.92, 1.0)) - 0.11;
    skin = smin(skin, sdCapsule(p, J[NECK], hc, 0.045), 0.03);
    for (int s = 0; s < 2; s++) {
        int e = s == 0 ? L_ELBOW : R_ELBOW, w = s == 0 ? L_WRIST : R_WRIST;
        vec3 hand = J[w] + 0.05 * normalize(J[w] - J[e]);
        float arm = sdRoundCone(p, J[e], J[w], 0.042, 0.033);
        arm = smin(arm, length(p - hand) - 0.042, 0.03);
        skin = min(skin, arm);
    }
    vec2 res = vec2(skin, 1.0);

    // shirt: chest and belly ellipsoids in the torso frame (wider than deep), shoulders, sleeves
    vec3 t_up = normalize(J[NECK] - J[PELVIS]);
    vec3 t_fwd = normalize(facing - t_up * dot(facing, t_up));
    mat3 torso = transpose(mat3(cross(t_up, t_fwd), t_up, t_fwd));
    float shirt = sdEllipsoid(torso * (p - mix(J[SPINE], J[NECK], 0.35)), vec3(0.17, 0.2, 0.115));
    shirt = smin(shirt, sdEllipsoid(torso * (p - mix(J[PELVIS], J[SPINE], 0.6)), vec3(0.14, 0.16, 0.1)), 0.06);
    shirt = smin(shirt, sdCapsule(p, J[L_SHOULDER], J[R_SHOULDER], 0.065), 0.06);
    shirt = smin(shirt, sdRoundCone(p, J[L_SHOULDER], J[L_ELBOW], 0.062, 0.05), 0.03);
    shirt = smin(shirt, sdRoundCone(p, J[R_SHOULDER], J[R_ELBOW], 0.062, 0.05), 0.03);
    add(res, shirt, 2.0, 0.015);

    // pants: hips, thighs, shins
    float pants = sdCapsule(p, J[L_HIP], J[R_HIP], 0.105);
    pants = smin(pants, sdRoundCone(p, J[L_HIP], J[L_KNEE], 0.085, 0.062), 0.04);
    pants = smin(pants, sdRoundCone(p, J[R_HIP], J[R_KNEE], 0.085, 0.062), 0.04);
    pants = smin(pants, sdRoundCone(p, J[L_KNEE], J[L_ANKLE], 0.062, 0.046), 0.02);
    pants = smin(pants, sdRoundCone(p, J[R_KNEE], J[R_ANKLE], 0.062, 0.046), 0.02);
    add(res, pants, 3.0, 0.01);

    // shoes
    float shoes = min(sdRoundCone(p, J[L_ANKLE] - vec3(0.0, 0.01, 0.0), J[L_TOE], 0.055, 0.042),
                      sdRoundCone(p, J[R_ANKLE] - vec3(0.0, 0.01, 0.0), J[R_TOE], 0.055, 0.042));
    add(res, shoes, 4.0, 0.01);

    // hair: a cap over the top and back of the head
    vec3 cut = normalize(facing * 0.8 - up * 0.6);
    float hair = max(length(p - hc - vec3(0.0, 0.015, 0.0)) - 0.12, dot(p - hc - up * 0.02, cut));
    add(res, hair, 5.0, 0.005);

    // eyes
    vec3 eye = hc + facing * 0.098 + up * 0.012;
    float eyes = min(length(p - eye - side * 0.038), length(p - eye + side * 0.038)) - 0.017;
    if (eyes < res.x) res = vec2(eyes, 6.0);
    return res;
}

vec3 calc_normal(vec3 p) {
    const vec2 k = vec2(1.0, -1.0);
    const float h = 5e-4;
    return normalize(k.xyy * map(p + k.xyy * h).x + k.yyx * map(p + k.yyx * h).x
                   + k.yxy * map(p + k.yxy * h).x + k.xxx * map(p + k.xxx * h).x);
}

float soft_shadow(vec3 ro, vec3 rd) {
    float res = 1.0, t = 0.02;
    for (int i = 0; i < 64; i++) {
        vec2 m = map(ro + rd * t);
        float h = m.x;
        // material 0 = the bounding-sphere shortcut, which underestimates the distance;
        // counting it as a near miss would draw a fake penumbra ring
        if (m.y > 0.5) res = min(res, 12.0 * h / t);
        t += clamp(h, 0.01, 0.2);
        if (res < 0.003 || t > 4.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

float ambient_occlusion(vec3 p, vec3 n) {
    float occ = 0.0, w = 1.0;
    for (int i = 1; i <= 5; i++) {
        float h = 0.04 * float(i);
        occ += w * (h - map(p + n * h).x);
        w *= 0.75;
    }
    return clamp(1.0 - 2.0 * occ, 0.0, 1.0);
}

vec3 albedo(float m) {
    if (m < 1.5) return vec3(0.93, 0.68, 0.53);  // skin
    if (m < 2.5) return shirt_color;
    if (m < 3.5) return vec3(0.13, 0.17, 0.3);   // pants
    if (m < 4.5) return vec3(0.9, 0.2, 0.12);    // shoes
    if (m < 5.5) return vec3(0.22, 0.13, 0.07);  // hair
    return vec3(0.03);                           // eyes
}

vec3 sky(vec3 rd) {
    return mix(vec3(0.75, 0.83, 0.92), vec3(0.25, 0.48, 0.88), clamp(rd.y * 2.5, 0.0, 1.0));
}

void main() {
    vec3 fwd = normalize(cam_target - cam_pos);
    vec3 right = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
    vec3 cup = cross(right, fwd);
    vec3 rd = normalize(fwd * 2.2 + right * uv.x + cup * uv.y);
    vec3 ro = cam_pos;
    vec3 sun = normalize(vec3(-0.6, 0.85, 0.35));
    vec3 sun_col = vec3(1.0, 0.92, 0.8);
    vec3 sky_col = vec3(0.55, 0.68, 0.95);

    float t_floor = rd.y < 0.0 ? -ro.y / rd.y : 1e9;
    float t = 0.0, mat = -1.0;
    for (int i = 0; i < 200; i++) {
        vec2 h = map(ro + rd * t);
        if (h.x < 5e-4 * max(t, 1.0)) { mat = h.y; break; }
        t += h.x;
        if (t > min(t_floor, 30.0)) break;
    }

    vec3 col;
    if (mat > 0.0 && t < t_floor) {
        vec3 p = ro + rd * t;
        vec3 n = calc_normal(p);
        vec3 alb = albedo(mat);
        float occ = ambient_occlusion(p, n);
        float dif = clamp(dot(n, sun), 0.0, 1.0);
        if (dif > 0.0) dif *= soft_shadow(p + n * 0.003, sun);
        float hemi = 0.5 + 0.5 * n.y;
        float bounce = clamp(-n.y, 0.0, 1.0);
        float rim = pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 3.0);
        float gloss = mat > 3.5 && mat < 4.5 ? 0.5 : (mat > 5.5 ? 1.0 : 0.15);
        float spe = pow(clamp(dot(reflect(rd, n), sun), 0.0, 1.0), 32.0) * dif * gloss;
        col = alb * (2.2 * dif * sun_col + 0.55 * hemi * occ * sky_col + 0.3 * bounce * occ * vec3(0.6, 0.55, 0.5));
        col += 0.35 * rim * occ * sky_col + spe;
    } else if (t_floor < 1e8) {
        vec3 p = ro + rd * t_floor;
        vec2 cell = floor(p.xz / 0.5);
        float checker = mod(cell.x + cell.y, 2.0);
        vec3 alb = mix(vec3(0.62, 0.64, 0.66), vec3(0.72, 0.74, 0.76), checker);
        vec2 f = abs(fract(p.xz / 0.5) - 0.5);
        alb *= 1.0 - 0.12 * smoothstep(0.47, 0.5, max(f.x, f.y));  // darker tile seams
        float sha = soft_shadow(p + vec3(0.0, 0.001, 0.0), sun);
        float occ = 0.35 + 0.65 * smoothstep(0.0, 0.2, map(p).x);  // contact shadow
        col = alb * (1.9 * sha * sun_col + 0.5 * occ * sky_col);
        col = mix(col, sky(rd), 1.0 - exp(-0.004 * t_floor * t_floor));
    } else {
        col = sky(rd);
    }
    col = 1.0 - exp(-col * 1.35);                   // soft tone map
    col = pow(col, vec3(0.4545));
    col *= 1.0 - 0.18 * dot(uv * 0.7, uv * 0.7);    // vignette
    frag_color = vec4(col, 1.0);
}
"""


def create_context():
    attempts = [{"backend": "egl"}, {}] if sys.platform.startswith("linux") else [{}]
    errors = []
    for kwargs in attempts:
        try:
            return moderngl.create_standalone_context(require=330, **kwargs)
        except Exception as e:
            errors.append(f"{kwargs.get('backend', 'native')}: {e}")
    raise RuntimeError("No OpenGL 3.3 context available (" + "; ".join(errors) + ")")


def facing_direction(frame):
    """Horizontal forward vector from the hips: left-hip direction x up."""
    hips = frame[NAMES.index("l_hip")] - frame[NAMES.index("r_hip")]
    fwd = np.cross(hips, [0.0, 1.0, 0.0])
    fwd[1] = 0.0
    return fwd / (np.linalg.norm(fwd) + 1e-8)


def render_clip(positions, source, size=480, aa=2):
    """positions (T, J, 3) -> list of PIL images, camera following the pelvis."""
    defines = "\n".join(f"#define {name.upper()} {i}" for i, name in enumerate(NAMES))
    shader = FRAGMENT_SHADER % {"num_joints": len(NAMES), "joint_defines": defines}
    ctx = create_context()
    try:
        prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=shader)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        vao = ctx.vertex_array(prog, [(ctx.buffer(quad.tobytes()), "2f", "in_pos")])
        px = size * aa
        fbo = ctx.simple_framebuffer((px, px), components=3)
        fbo.use()
        prog["shirt_color"].value = SHIRT.get(source, SHIRT["procedural"])
        offset = np.array([-2.1, 0.45, 2.45])  # third-person camera, front-right of the character
        frames = []
        for frame in positions:
            target = np.array([frame[0, 0], 1.05, frame[0, 2]])
            prog["J"].write(frame.astype("f4").tobytes())
            prog["cam_target"].value = tuple(target)
            prog["cam_pos"].value = tuple(target + offset)
            prog["facing"].value = tuple(facing_direction(frame))
            fbo.clear()
            vao.render(moderngl.TRIANGLE_STRIP)
            img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(px, px, 3)
            frames.append(Image.fromarray(np.flipud(img)).resize((size, size), Image.LANCZOS))
        return frames
    finally:
        ctx.release()


def label(img, text):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(14, img.width // 24))
    draw.rectangle((0, 0, img.width, font.size + 14), fill=(20, 22, 28))
    draw.text((10, 6), text, fill=(240, 240, 240), font=font)
    return img


def save_animation(frames, stem, fps, gif=False):
    path = stem + (".gif" if gif else ".webp")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    extra = {} if gif else {"quality": 80, "method": 4}
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0, **extra)
    print(f"Saved {path}")


def render_file(path, size, gif=False):
    clip = np.load(path)
    source = str(clip["source"])
    frames = render_clip(clip["positions"], source, size)
    frames = [label(f, f"{LABELS.get(source, source)}: \"{clip['text']}\"") for f in frames]
    save_animation(frames, os.path.join(OUT_DIR, os.path.splitext(os.path.basename(path))[0]), int(clip["fps"]), gif)


def render_compare(command, size, gif=False):
    """procedural | AR | DDPM side by side for one command."""
    paths = [os.path.join(HERE, "outputs", f"{command}_{source}.npz") for source in ("procedural", "ar", "diffusion")]
    render_grid([p for p in paths if os.path.exists(p)], f"{command}_compare", size, gif, with_text=False)


def render_grid(paths, name, size, gif=False, with_text=True):
    """The given clips side by side in one animation."""
    panels, fps = [], 30
    for path in paths:
        clip = np.load(path)
        source, fps = str(clip["source"]), int(clip["fps"])
        title = LABELS.get(source, source) + (f": {clip['text']}" if with_text else "")
        panels.append([label(f, title) for f in render_clip(clip["positions"], source, size)])
    frames = []
    for row in zip(*panels):
        sheet = Image.new("RGB", (size * len(row), size))
        for i, img in enumerate(row):
            sheet.paste(img, (i * size, 0))
        frames.append(sheet)
    save_animation(frames, os.path.join(OUT_DIR, name), fps, gif)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="*", help=".npz files from generate_motion.py")
    parser.add_argument("--compare", action="store_true", help="side-by-side animation per command in outputs/")
    parser.add_argument("--grid", nargs="+", metavar="NPZ", help="render these clips side by side")
    parser.add_argument("--out", default="grid", help="output name for --grid (default: grid)")
    parser.add_argument("--size", type=int, default=400, help="pixels per panel (default 400)")
    parser.add_argument("--gif", action="store_true", help="write GIF instead of animated WebP")
    args = parser.parse_args()
    if not (args.clips or args.compare or args.grid):
        parser.error("give .npz clips, --compare or --grid")
    if args.grid:
        render_grid(args.grid, args.out, args.size, args.gif)
    for path in args.clips:
        render_file(path, args.size, args.gif)
    if args.compare:
        commands = sorted({os.path.basename(p).rsplit("_", 1)[0]
                           for p in glob.glob(os.path.join(HERE, "outputs", "*_procedural.npz"))})
        for command in commands:
            render_compare(command, args.size, args.gif)


if __name__ == "__main__":
    main()
