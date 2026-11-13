// A few SDFs in C++, mirroring sdf_lib.glsl (shared by the demo and tests).
#pragma once
#include <algorithm>
#include <cmath>

namespace vibecad::shapes {

inline float sphere(float x, float y, float z, float r) { return std::sqrt(x * x + y * y + z * z) - r; }

// Ring in the XZ plane: ring radius R, tube radius r.
inline float torus(float x, float y, float z, float R, float r) {
    const float q = std::sqrt(x * x + z * z) - R;
    return std::sqrt(q * q + y * y) - r;
}

inline float smooth_union(float a, float b, float k) {
    const float h = std::clamp(0.5f + 0.5f * (b - a) / k, 0.f, 1.f);
    return b + (a - b) * h - k * h * (1.f - h);
}

// Three smoothly blended spheres, like examples/snowman.glsl.
inline float snowman(float x, float y, float z) {
    const float base = sphere(x, y + 0.55f, z, 0.45f);
    const float torso = sphere(x, y - 0.15f, z, 0.33f);
    const float head = sphere(x, y - 0.62f, z, 0.23f);
    return smooth_union(smooth_union(base, torso, 0.08f), head, 0.06f);
}

}  // namespace vibecad::shapes
