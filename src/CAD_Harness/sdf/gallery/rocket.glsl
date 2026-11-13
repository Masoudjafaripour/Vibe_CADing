// Prompt: "A retro rocket standing on three swept fins, with a round porthole and a nozzle"
// LLM: Claude (Opus 5.5), written from the prompt above

const float TAU = 6.2831853;

float body(vec3 p) {
    float hull = sdRoundCone(p - vec3(0.0, -0.62, 0.0), 0.2, 0.17, 0.95);
    float nose = sdRoundCone(p - vec3(0.0, 0.33, 0.0), 0.17, 0.015, 0.55);
    return opSmoothUnion(hull, nose, 0.12);
}

float fins(vec3 p) {
    float sector = TAU / 3.0;
    float a = mod(atan(p.z, p.x) + sector * 0.5, sector) - sector * 0.5;
    vec3 q = vec3(length(p.xz) * cos(a), p.y, length(p.xz) * sin(a));
    // Swept fin: a box sheared backwards and clipped by a slanted leading edge
    vec3 f = q - vec3(0.32, -0.72, 0.0);
    f.y -= 0.6 * f.x;  // shear: fins sweep down and back
    return (sdBox(f, vec3(0.16, 0.2, 0.018)) - 0.01) * 0.85;
}

float nozzle(vec3 p) {
    float bell = sdRoundCone(p - vec3(0.0, -0.8, 0.0), 0.13, 0.09, 0.14);
    return max(bell, -sdCylinder(p - vec3(0.0, -0.86, 0.0), 0.09, 0.1));
}

float window(vec3 p) {
    vec3 q = rotX(p - vec3(0.0, 0.08, 0.18), 1.5708);
    return sdTorus(q, 0.07, 0.022);
}

float sdf(vec3 p) {
    return min(min(opSmoothUnion(body(p), fins(p), 0.04), nozzle(p)), window(p));
}

vec3 sdf_color(vec3 p) {
    float d = sdf(p) + 2e-3;
    if (window(p) <= d) return vec3(0.85, 0.65, 0.2);
    if (nozzle(p) <= d) return vec3(0.3, 0.3, 0.33);
    if (fins(p) <= d || p.y > 0.45) return vec3(0.85, 0.15, 0.12);
    if (abs(p.x) < 0.2 && p.z > 0.12 && length(p.xy - vec2(0.0, 0.08)) < 0.06) return vec3(0.25, 0.45, 0.7);
    return vec3(0.93, 0.93, 0.9);
}
