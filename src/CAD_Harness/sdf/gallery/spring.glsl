// Prompt: "A compression spring: 5 coils of round wire with flat ground ends"
// LLM: Claude (Opus 5.5), written from the prompt above

const float TAU = 6.2831853;

float sdf(vec3 p) {
    const float R = 0.42;      // coil radius
    const float wire = 0.075;  // wire radius
    const float pitch = 0.33;  // rise per turn
    vec3 q = p - vec3(0.0, -0.1, 0.0);

    // Height along the helix, wrapped to the nearest turn
    float y = q.y - pitch * atan(q.z, q.x) / TAU;
    y = mod(y + 0.5 * pitch, pitch) - 0.5 * pitch;
    float coil = (length(vec2(length(q.xz) - R, y)) - wire) * 0.8;  // 0.8: helix is not an exact distance

    // Flat ground ends
    return max(coil, abs(q.y) - 0.86);
}

vec3 sdf_color(vec3 p) {
    return vec3(0.7, 0.72, 0.76);
}
