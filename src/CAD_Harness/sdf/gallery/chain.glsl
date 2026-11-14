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
    // Links tilted +-45 degrees about the chain axis: still interlocked (90 degrees
    // apart), and every link touches the build plate, so none prints in mid-air.
    const float lift = (R + r) * 0.70710678;
    vec3 c = rotY(p - vec3(0.0, -1.0 + lift, 0.0), -0.75);
    float outer = min(link(rotX(c + vec3(STEP, 0.0, 0.0), 0.785398), L, R, r),
                      link(rotX(c - vec3(STEP, 0.0, 0.0), 0.785398), L, R, r));
    float middle = link(rotX(c, -0.785398), L, R, r);
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
