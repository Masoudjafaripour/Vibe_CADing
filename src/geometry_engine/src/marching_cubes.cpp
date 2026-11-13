#include <unordered_map>

#include "vibecad/mc_tables.hpp"
#include "vibecad/mesh.hpp"

namespace vibecad {
namespace {

// Cube corners (Bourke's numbering) and the 12 edges as corner pairs.
constexpr int kCorner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                               {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
constexpr int kEdge[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                              {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

}  // namespace

Mesh marching_cubes(const Grid& g, float iso) {
    Mesh mesh;
    const int n = g.n;
    const uint64_t m = n + 1;
    const float h = g.step();

    // Each lattice edge holds at most one vertex, shared by the up-to-4 cells
    // around it. Key = (index of the edge's lower endpoint) * 3 + axis.
    std::unordered_map<uint64_t, uint32_t> edge_vertex;
    edge_vertex.reserve(size_t(n) * n * 8);

    auto vertex_on_edge = [&](const int (&lo)[3], int axis) -> uint32_t {
        const uint64_t key = ((lo[2] * m + lo[1]) * m + lo[0]) * 3 + axis;
        auto [it, inserted] = edge_vertex.try_emplace(key, uint32_t(mesh.vertices.size()));
        if (inserted) {
            // Interpolate from the lower endpoint so every cell computes the
            // same point for a shared edge.
            int hi[3] = {lo[0], lo[1], lo[2]};
            ++hi[axis];
            const float a = g.at(lo[0], lo[1], lo[2]), b = g.at(hi[0], hi[1], hi[2]);
            const float t = (iso - a) / (b - a);
            Vec3 p{g.lo + lo[0] * h, g.lo + lo[1] * h, g.lo + lo[2] * h};
            p[axis] += t * h;
            mesh.vertices.push_back(p);
        }
        return it->second;
    };

    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                int mask = 0;
                for (int c = 0; c < 8; ++c)
                    if (g.at(i + kCorner[c][0], j + kCorner[c][1], k + kCorner[c][2]) < iso) mask |= 1 << c;
                if (mask == 0 || mask == 255) continue;  // cell fully outside or inside

                uint32_t ids[12];
                const int8_t* row = kTriTable[mask];
                for (int t = 0; row[t] != -1; ++t) {
                    const int e = row[t];
                    const int* ca = kCorner[kEdge[e][0]];
                    const int* cb = kCorner[kEdge[e][1]];
                    int lo[3], axis = 0;
                    for (int d = 0; d < 3; ++d) {
                        lo[d] = (d == 0 ? i : d == 1 ? j : k) + std::min(ca[d], cb[d]);
                        if (ca[d] != cb[d]) axis = d;
                    }
                    ids[e] = vertex_on_edge(lo, axis);
                }
                for (int t = 0; row[t] != -1; t += 3)
                    // Bourke's tables wind clockwise from outside; swap to CCW.
                    mesh.triangles.push_back({ids[row[t]], ids[row[t + 2]], ids[row[t + 1]]});
            }
    return mesh;
}

}  // namespace vibecad
