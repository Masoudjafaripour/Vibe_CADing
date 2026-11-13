#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "vibecad/mesh.hpp"

namespace vibecad {

void write_stl(const Mesh& mesh, const std::string& path) {
    // Layout: 80-byte header, uint32 count, then 50 bytes per triangle
    // (normal + 3 vertices as float32, uint16 attribute). One buffered write.
    constexpr size_t kTri = 50;
    std::vector<char> buf(84 + kTri * mesh.triangles.size(), 0);
    std::strncpy(buf.data(), "Vibe-CADing geometry_engine", 80);
    const uint32_t count = uint32_t(mesh.triangles.size());
    std::memcpy(buf.data() + 80, &count, 4);

    char* p = buf.data() + 84;
    for (const auto& t : mesh.triangles) {
        const Vec3 &a = mesh.vertices[t[0]], &b = mesh.vertices[t[1]], &c = mesh.vertices[t[2]];
        const float u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const float v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        float n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
        const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len > 0) for (float& x : n) x /= len;
        std::memcpy(p, n, 12);
        std::memcpy(p + 12, a.data(), 12);
        std::memcpy(p + 24, b.data(), 12);
        std::memcpy(p + 36, c.data(), 12);
        p += kTri;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open " + path);
    out.write(buf.data(), std::streamsize(buf.size()));
}

}  // namespace vibecad
