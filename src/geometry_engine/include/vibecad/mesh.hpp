// SDF -> triangle mesh: sampling, marching cubes, mesh checks, STL export.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vibecad {

using Vec3 = std::array<float, 3>;

// Signed distances sampled on an (n+1)^3 lattice spanning [lo, hi]^3.
struct Grid {
    int n = 0;  // cells per axis
    float lo = -1.f, hi = 1.f;
    std::vector<float> values;  // x fastest: index (k*(n+1) + j)*(n+1) + i

    float step() const { return (hi - lo) / n; }
    float at(int i, int j, int k) const {
        return values[(size_t(k) * (n + 1) + j) * (n + 1) + i];
    }
};

// Indexed triangle mesh: neighbouring triangles share vertex indices.
struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<std::array<uint32_t, 3>> triangles;  // counter-clockwise from outside
};

struct MeshStats {
    double volume = 0, area = 0;
    long edges = 0;
    long components = 0;     // separate closed shells
    long euler = 0;          // V - E + F
    long genus = 0;          // handles summed over shells (sphere 0, torus 1)
    bool watertight = false; // every edge is shared by exactly two triangles
    bool oriented = false;   // ...which traverse it in opposite directions
};

// Evaluates sdf(x, y, z) at every lattice point (parallel when built with OpenMP).
template <class SDF>
Grid sample(SDF&& sdf, int n, float lo = -1.f, float hi = 1.f) {
    const int m = n + 1;
    Grid g{n, lo, hi, std::vector<float>(size_t(m) * m * m)};
    const float h = g.step();
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < m; ++k)
        for (int j = 0; j < m; ++j) {
            float* row = &g.values[(size_t(k) * m + j) * m];
            for (int i = 0; i < m; ++i) row[i] = sdf(lo + i * h, lo + j * h, lo + k * h);
        }
    return g;
}

// Extracts the iso-surface sdf = iso; the inside (sdf < iso) is enclosed.
Mesh marching_cubes(const Grid& grid, float iso = 0.f);

MeshStats analyze(const Mesh& mesh);

// Binary STL (little-endian), readable by any slicer or mesh viewer.
void write_stl(const Mesh& mesh, const std::string& path);

}  // namespace vibecad
