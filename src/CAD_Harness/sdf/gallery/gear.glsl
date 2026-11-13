// Prompt: "A spur gear with 16 teeth, a raised hub with a keyed bore, and six round lightening holes"
// LLM: Claude (Opus 5.5), written from the prompt above

const float TAU = 6.2831853;

float box2(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// Rotates q into the first of n equal sectors (polar repetition).
vec2 polar_repeat(vec2 q, float n) {
    float sector = TAU / n;
    float a = mod(atan(q.y, q.x) + sector * 0.5, sector) - sector * 0.5;
    return length(q) * vec2(cos(a), sin(a));
}

// Extrudes a 2D distance along y with half-height h.
float extrude(float d2, float y, float h) {
    vec2 w = vec2(d2, abs(y) - h);
    return min(max(w.x, w.y), 0.0) + length(max(w, 0.0));
}

float gear2d(vec2 q) {
    float rim = length(q) - 0.72;
    vec2 t = polar_repeat(q, 16.0);
    float tooth = box2(t - vec2(0.76, 0.0), vec2(0.09, 0.05 - 0.12 * (t.x - 0.72) * 0.5)) - 0.01;
    float holes = length(polar_repeat(q, 6.0) - vec2(0.46, 0.0)) - 0.11;
    return max(min(rim, tooth), -holes);
}

float bore(vec2 q) {
    float hole = length(q) - 0.11;
    float key = box2(q - vec2(0.11, 0.0), vec2(0.04, 0.035));
    return min(hole, key);
}

float sdf(vec3 p) {
    vec3 q = p - vec3(0.0, -0.88, 0.0);
    float plate = extrude(gear2d(q.xz), q.y, 0.08);
    float hub = extrude(length(q.xz) - 0.22, q.y - 0.05, 0.13);
    return max(min(plate, hub), -extrude(bore(q.xz), q.y, 0.5));
}

vec3 sdf_color(vec3 p) {
    float r = length(p.xz);
    return r < 0.24 ? vec3(0.62, 0.64, 0.68) : vec3(0.78, 0.6, 0.25);
}
