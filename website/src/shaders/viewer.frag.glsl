#version 300 es
// WebGL2 port of the sphere tracer in src/CAD_Harness/sdf/render.py.
// sdf_lib.glsl and the shape's code are spliced in at the marker below.
precision highp float;
precision highp int;
in vec2 uv;
out vec4 frag_color;
uniform vec3 cam_pos;
uniform float aspect;
uniform bool shadows;
uniform vec3 bg_top;
uniform vec3 bg_bottom;

//__SDF_CODE__

const float FLOOR_Y = -1.0;

vec3 calc_normal(vec3 p) {
    const vec2 k = vec2(1.0, -1.0);
    const float h = 5e-4;
    return normalize(k.xyy * sdf(p + k.xyy * h) + k.yyx * sdf(p + k.yyx * h)
                   + k.yxy * sdf(p + k.yxy * h) + k.xxx * sdf(p + k.xxx * h));
}

float soft_shadow(vec3 ro, vec3 rd) {
    float res = 1.0;
    float t = 0.02;
    for (int i = 0; i < 64; i++) {
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
    vec3 rd = normalize(fwd * 2.0 + right * uv.x * aspect + up * uv.y);
    vec3 ro = cam_pos;
    vec3 bg = mix(bg_bottom, bg_top, 0.5 + 0.5 * uv.y);

    int hit = 0;
    float t = 0.0;
    for (int i = 0; i < 300; i++) {
        vec3 p = ro + rd * t;
        float d_obj = sdf(p);
        float d_floor = p.y - FLOOR_Y;
        float d = min(d_obj, d_floor);
        if (d < 1e-4 * max(t, 1.0)) { hit = d_obj < d_floor ? 1 : 2; break; }
        t += d * 0.9;
        if (t > 20.0) break;
    }

    vec3 col = pow(bg, vec3(2.2));
    vec3 p = ro + rd * t;
    if (hit == 1) {
        col = shade(p, calc_normal(p), rd, base_color(p));
    } else if (hit == 2) {
        col = shade(p, vec3(0.0, 1.0, 0.0), rd, vec3(0.5));
        col = mix(col, pow(bg, vec3(2.2)), 1.0 - exp(-0.02 * t * t));
    }
    frag_color = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.4545)), 1.0);
}
