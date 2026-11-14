// Prompt: "A twisted vase with eight flowing ridges, a curvy profile, and an open top"
// LLM: Claude (Opus 5.5), written from the prompt above

float sdf(vec3 p) {
    float t = clamp((p.y + 1.0) / 1.85, 0.0, 1.0);          // 0 at the base, 1 at the lip
    float profile = 0.28 + 0.2 * sin(3.1 * t + 0.25) - 0.08 * t;  // bulge, then narrow neck
    float angle = atan(p.z, p.x) + 2.2 * t;                   // twist with height
    float ridged = profile * (1.0 + 0.09 * cos(8.0 * angle));

    float outer = (length(p.xz) - ridged) * 0.6;              // 0.6: twist distorts distances
    float wall = abs(outer) - 0.025;
    float solid_base = max(outer, p.y + 0.92);                // closed bottom
    float body = min(wall, solid_base);
    body = max(body, -1.0 - p.y);                             // nothing below the floor
    return max(body, p.y - 0.85);                             // open top
}

vec3 sdf_color(vec3 p) {
    float t = clamp((p.y + 1.0) / 1.85, 0.0, 1.0);
    return mix(vec3(0.55, 0.2, 0.45), vec3(0.95, 0.6, 0.35), t);
}
