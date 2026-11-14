// Taubin smoothing, vertex normals and binary glTF (.glb) export.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "vibecad/mesh_ops.hpp"

namespace vibecad {
namespace {

// Unique neighbours of every vertex, as a compact CSR list.
void neighbours(const Mesh& mesh, std::vector<uint32_t>& offset, std::vector<uint32_t>& adj) {
    const size_t n = mesh.vertices.size();
    std::vector<std::vector<uint32_t>> lists(n);
    for (const auto& t : mesh.triangles)
        for (int e = 0; e < 3; ++e) lists[t[e]].push_back(t[(e + 1) % 3]), lists[t[(e + 1) % 3]].push_back(t[e]);
    offset.assign(n + 1, 0);
    adj.clear();
    for (size_t v = 0; v < n; ++v) {
        auto& l = lists[v];
        std::sort(l.begin(), l.end());
        l.erase(std::unique(l.begin(), l.end()), l.end());
        adj.insert(adj.end(), l.begin(), l.end());
        offset[v + 1] = uint32_t(adj.size());
    }
}

}  // namespace

void smooth_taubin(Mesh& mesh, int iterations, float lambda, float mu) {
    std::vector<uint32_t> offset, adj;
    neighbours(mesh, offset, adj);
    auto& V = mesh.vertices;
    std::vector<Vec3> next(V.size());
    auto step = [&](float factor) {
#pragma omp parallel for schedule(static)
        for (long v = 0; v < long(V.size()); ++v) {
            const uint32_t b = offset[v], e = offset[v + 1];
            if (b == e) {
                next[v] = V[v];
                continue;
            }
            Vec3 avg{0, 0, 0};
            for (uint32_t i = b; i < e; ++i)
                for (int d = 0; d < 3; ++d) avg[d] += V[adj[i]][d];
            for (int d = 0; d < 3; ++d) next[v][d] = V[v][d] + factor * (avg[d] / float(e - b) - V[v][d]);
        }
        V.swap(next);
    };
    for (int i = 0; i < iterations; ++i) {
        step(lambda);  // shrink (smooth)...
        step(mu);      // ...then inflate back: removes noise without shrinking
    }
}

std::vector<Vec3> vertex_normals(const Mesh& mesh) {
    std::vector<Vec3> n(mesh.vertices.size(), Vec3{0, 0, 0});
    for (const auto& t : mesh.triangles) {
        const Vec3 &a = mesh.vertices[t[0]], &b = mesh.vertices[t[1]], &c = mesh.vertices[t[2]];
        const float u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, w[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const Vec3 f{u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};  // |f| = 2 area
        for (uint32_t i : t)
            for (int d = 0; d < 3; ++d) n[i][d] += f[d];
    }
    for (Vec3& v : n) {
        const float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (len > 0) for (float& x : v) x /= len;
        else v = {0, 1, 0};
    }
    return n;
}

void write_glb(const Mesh& mesh, const std::string& path, const std::vector<Vec3>& colors, const std::string& name) {
    const size_t nv = mesh.vertices.size(), ni = mesh.triangles.size() * 3;
    const bool has_color = !colors.empty();
    if (has_color && colors.size() != nv) throw std::invalid_argument("colors must have one entry per vertex");
    const std::vector<Vec3> normals = vertex_normals(mesh);

    // Binary buffer: positions | normals | colours (RGBA8) | indices; every view 4-byte aligned.
    const size_t pos_bytes = nv * 12, col_bytes = has_color ? nv * 4 : 0, idx_bytes = ni * 4;
    std::vector<char> bin(pos_bytes * 2 + col_bytes + idx_bytes);
    std::memcpy(bin.data(), mesh.vertices.data(), pos_bytes);
    std::memcpy(bin.data() + pos_bytes, normals.data(), pos_bytes);
    for (size_t v = 0; v < nv && has_color; ++v)
        for (int d = 0; d < 4; ++d)
            bin[pos_bytes * 2 + v * 4 + d] =
                char(d == 3 ? 255 : int(std::lround(std::clamp(colors[v][d], 0.f, 1.f) * 255)));
    std::memcpy(bin.data() + pos_bytes * 2 + col_bytes, mesh.triangles.data(), idx_bytes);

    Vec3 lo{0, 0, 0}, hi{0, 0, 0};
    if (nv) lo = hi = mesh.vertices[0];
    for (const Vec3& v : mesh.vertices)
        for (int d = 0; d < 3; ++d) lo[d] = std::min(lo[d], v[d]), hi[d] = std::max(hi[d], v[d]);

    std::ostringstream js;
    js.precision(9);
    const int idx_accessor = has_color ? 3 : 2;
    js << R"({"asset":{"version":"2.0","generator":"Vibe-CADing geometry_engine"},"scene":0,"scenes":[{"nodes":[0]}],)"
       << R"("nodes":[{"mesh":0,"name":")" << name << R"("}],"meshes":[{"name":")" << name
       << R"(","primitives":[{"attributes":{"POSITION":0,"NORMAL":1)" << (has_color ? R"(,"COLOR_0":2)" : "")
       << R"(},"indices":)" << idx_accessor << R"(,"material":0,"mode":4}]}],)"
       << R"("materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1],"metallicFactor":0.0,"roughnessFactor":0.55}}],)"
       << R"("buffers":[{"byteLength":)" << bin.size() << "}],\"bufferViews\":["
       << R"({"buffer":0,"byteOffset":0,"byteLength":)" << pos_bytes << R"(,"target":34962},)"
       << R"({"buffer":0,"byteOffset":)" << pos_bytes << R"(,"byteLength":)" << pos_bytes << R"(,"target":34962},)";
    if (has_color)
        js << R"({"buffer":0,"byteOffset":)" << pos_bytes * 2 << R"(,"byteLength":)" << col_bytes << R"(,"target":34962},)";
    js << R"({"buffer":0,"byteOffset":)" << pos_bytes * 2 + col_bytes << R"(,"byteLength":)" << idx_bytes
       << R"(,"target":34963}],"accessors":[)"
       << R"({"bufferView":0,"componentType":5126,"count":)" << nv << R"(,"type":"VEC3","min":[)" << lo[0] << ","
       << lo[1] << "," << lo[2] << "],\"max\":[" << hi[0] << "," << hi[1] << "," << hi[2] << "]},"
       << R"({"bufferView":1,"componentType":5126,"count":)" << nv << R"(,"type":"VEC3"},)";
    if (has_color)
        js << R"({"bufferView":2,"componentType":5121,"normalized":true,"count":)" << nv << R"(,"type":"VEC4"},)";
    js << R"({"bufferView":)" << idx_accessor << R"(,"componentType":5125,"count":)" << ni << R"(,"type":"SCALAR"}]})";

    std::string json = js.str();
    while (json.size() % 4) json += ' ';  // chunks are 4-byte aligned (JSON pads with spaces)
    while (bin.size() % 4) bin.push_back(0);

    auto u32 = [](std::ofstream& o, uint32_t x) { o.write(reinterpret_cast<const char*>(&x), 4); };
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open " + path);
    u32(out, 0x46546C67);  // "glTF"
    u32(out, 2);
    u32(out, uint32_t(12 + 8 + json.size() + 8 + bin.size()));
    u32(out, uint32_t(json.size()));
    u32(out, 0x4E4F534A);  // "JSON"
    out.write(json.data(), std::streamsize(json.size()));
    u32(out, uint32_t(bin.size()));
    u32(out, 0x004E4942);  // "BIN\0"
    out.write(bin.data(), std::streamsize(bin.size()));
}

}  // namespace vibecad
