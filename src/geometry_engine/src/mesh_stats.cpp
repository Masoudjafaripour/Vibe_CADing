#include <cmath>
#include <numeric>
#include <unordered_map>

#include "vibecad/mesh.hpp"

namespace vibecad {
namespace {

Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double dot(const Vec3& a, const Vec3& b) { return double(a[0]) * b[0] + double(a[1]) * b[1] + double(a[2]) * b[2]; }

uint32_t find(std::vector<uint32_t>& parent, uint32_t x) {
    while (parent[x] != x) x = parent[x] = parent[parent[x]];  // path halving
    return x;
}

}  // namespace

MeshStats analyze(const Mesh& mesh) {
    MeshStats s;
    std::unordered_map<uint64_t, int> directed;  // (from << 32 | to) -> count
    directed.reserve(mesh.triangles.size() * 3);
    std::vector<uint32_t> parent(mesh.vertices.size());
    std::iota(parent.begin(), parent.end(), 0u);

    for (const auto& t : mesh.triangles) {
        const Vec3 &a = mesh.vertices[t[0]], &b = mesh.vertices[t[1]], &c = mesh.vertices[t[2]];
        const Vec3 n = cross(sub(b, a), sub(c, a));
        s.area += 0.5 * std::sqrt(dot(n, n));
        s.volume += dot(a, cross(b, c)) / 6.0;  // signed tetrahedron with the origin
        for (int e = 0; e < 3; ++e) {
            const uint32_t u = t[e], v = t[(e + 1) % 3];
            ++directed[uint64_t(u) << 32 | v];
            parent[find(parent, u)] = find(parent, v);
        }
    }

    // Closed + consistently oriented: each directed edge appears once, and so does its reverse.
    s.watertight = s.oriented = !mesh.triangles.empty();
    for (const auto& [key, count] : directed) {
        const uint64_t reverse = (key << 32) | (key >> 32);
        const auto it = directed.find(reverse);
        const int back = it == directed.end() ? 0 : it->second;
        if (count + back != 2) s.watertight = false;
        if (count != 1 || back != 1) s.oriented = false;
    }

    s.edges = 0;
    for (const auto& [key, count] : directed)
        if ((key >> 32) < (key & 0xffffffffu) || !directed.count((key << 32) | (key >> 32))) ++s.edges;
    for (uint32_t v = 0; v < parent.size(); ++v)
        if (find(parent, v) == v) ++s.components;
    s.euler = long(mesh.vertices.size()) - s.edges + long(mesh.triangles.size());
    s.genus = (2 * s.components - s.euler) / 2;  // Euler-Poincare, per closed shell
    return s;
}

}  // namespace vibecad
