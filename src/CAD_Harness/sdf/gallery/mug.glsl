// Prompt: "A ceramic coffee mug with a curved handle, filled with coffee"
// LLM: Claude (Opus 5.5), written from the prompt above

float cup(vec3 p) {
    vec3 q = p - vec3(-0.1, -0.5, 0.0);
    float outer = sdCylinder(q, 0.45, 0.5) - 0.03;
    float inner = sdCylinder(q - vec3(0.0, 0.12, 0.0), 0.39, 0.5);
    return max(outer, -inner);
}

float handle(vec3 p) {
    // Torus in the XY plane, only the half outside the cup
    vec3 q = rotX(p - vec3(0.36, -0.45, 0.0), 1.5708);
    float ring = sdTorus(q, 0.22, 0.055);
    return max(ring, 0.3 - p.x);
}

float coffee(vec3 p) {
    return sdCylinder(p - vec3(-0.1, -0.5, 0.0), 0.4, 0.4);  // overlaps the walls and floor: one solid print
}

float sdf(vec3 p) {
    return min(opSmoothUnion(cup(p), handle(p), 0.04), coffee(p));
}

vec3 sdf_color(vec3 p) {
    if (coffee(p) < cup(p) - 1e-3) return vec3(0.22, 0.12, 0.06);
    return vec3(0.18, 0.42, 0.62);
}
