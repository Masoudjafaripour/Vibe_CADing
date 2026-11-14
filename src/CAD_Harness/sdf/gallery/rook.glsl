// Prompt: "A chess rook: wide round base, tapered body, and a hollow crenellated tower top"
// LLM: Claude (Opus 5.5), written from the prompt above

const float TAU = 6.2831853;

float sdf(vec3 p) {
    float r = length(p.xz);
    // Lathe profile, bottom to top
    float base = sdCylinder(p - vec3(0.0, -0.93, 0.0), 0.42, 0.07) - 0.02;
    float collar = sdTorus(p - vec3(0.0, -0.82, 0.0), 0.33, 0.06);
    // Starts inside the base, high enough that neither its rounded end nor the smooth
    // blend bulges below the flat foot (the foot must sit flat on the plate)
    float body = sdRoundCone(p - vec3(0.0, -0.6, 0.0), 0.26, 0.2, 0.75);
    float neck = sdTorus(p - vec3(0.0, 0.17, 0.0), 0.22, 0.045);
    float crown = sdCylinder(p - vec3(0.0, 0.36, 0.0), 0.3, 0.18) - 0.015;

    float d = opSmoothUnion(base, collar, 0.05);
    d = opSmoothUnion(d, body, 0.08);
    d = opSmoothUnion(d, neck, 0.04);
    d = opSmoothUnion(d, crown, 0.05);

    // Hollow top and six notches (crenellations)
    float hollow = sdCylinder(p - vec3(0.0, 0.5, 0.0), 0.2, 0.2);
    float sector = TAU / 6.0;
    float a = mod(atan(p.z, p.x) + sector * 0.5, sector) - sector * 0.5;
    vec3 q = vec3(r * cos(a), p.y, r * sin(a));
    float notch = sdBox(q - vec3(0.3, 0.52, 0.0), vec3(0.15, 0.08, 0.07));
    return max(d, -min(hollow, notch));
}

vec3 sdf_color(vec3 p) {
    return mix(vec3(0.12, 0.1, 0.09), vec3(0.25, 0.2, 0.17), smoothstep(-1.0, 0.6, p.y));
}
