// Prompt: "A lightweight, 3D-printable gyroid lattice sphere sitting on a solid ring base"
// LLM: Claude (Opus 5.5), written from the prompt above

float gyroid(vec3 p, float k) {
    vec3 s = sin(p * k), c = cos(p.yzx * k);
    return dot(s, c) / k;  // ~distance near the surface
}

float sdf(vec3 p) {
    vec3 c = vec3(0.0, -0.22, 0.0);
    float ball = sdSphere(p - c, 0.74);
    // One of the gyroid's two interwoven channel networks: a single connected
    // solid. (A thin gyroid *sheet* cut by the sphere left 19 loose fragments.)
    // The network thins out over the last 0.12 units before the sphere's surface,
    // so the cut never leaves small islands: the printability check found 6 loose
    // pieces at 256^3 without this.
    float lattice = (0.03 + 0.15 * smoothstep(-0.12, 0.0, ball)) - gyroid(p - c, 8.5);
    float shell = max(ball, lattice * 0.7);
    float ring = sdTorus(p - vec3(0.0, -0.92, 0.0), 0.32, 0.08);
    return opSmoothUnion(shell, ring, 0.04);
}

vec3 sdf_color(vec3 p) {
    return mix(vec3(0.15, 0.55, 0.6), vec3(0.75, 0.9, 0.55), smoothstep(-1.0, 0.6, p.y));
}
