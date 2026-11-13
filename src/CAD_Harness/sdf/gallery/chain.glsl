// Prompt: "Three interlocked chain links that can be 3D-printed in one piece without supports"
// LLM: Claude (Opus 5.5), written from the prompt above

// Stadium-shaped link in the XY plane: straight half-length L, ring radius R, wire r.
float link(vec3 p, float L, float R, float r) {
    vec3 q = vec3(max(abs(p.x) - L, 0.0), p.y, p.z);
    return length(vec2(length(q.xy) - R, q.z)) - r;
}

const float L = 0.2, R = 0.2, r = 0.07, STEP = 0.65;

// Distances to the outer links (x) and the middle link (y).
vec2 links(vec3 p) {
    // Lie along the box diagonal (more room), resting on the floor
    vec3 c = rotY(p - vec3(0.0, -1.0 + R + r, 0.0), -0.75);
    float outer = min(link(c + vec3(STEP, 0.0, 0.0), L, R, r), link(c - vec3(STEP, 0.0, 0.0), L, R, r));
    // Middle link turned 90 degrees about X so it threads through its neighbours
    float middle = link(rotX(c, 1.5708), L, R, r);
    return vec2(outer, middle);
}

float sdf(vec3 p) {
    vec2 d = links(p);
    return min(d.x, d.y);
}

vec3 sdf_color(vec3 p) {
    vec2 d = links(p);
    return d.y < d.x ? vec3(0.85, 0.65, 0.25) : vec3(0.65, 0.67, 0.72);
}
