// Checks marching cubes against shapes with known volume and topology.
// A wrong table entry would show up as a hole, a flipped face or bad genus.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "vibecad/mesh.hpp"
#include "vibecad/shapes.hpp"

using namespace vibecad;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? "pass" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

double rel_err(double got, double want) { return std::abs(got - want) / want; }

template <class SDF>
MeshStats mesh_of(SDF&& sdf, int n) { return analyze(marching_cubes(sample(sdf, n))); }

}  // namespace

int main() {
    const double pi = std::acos(-1.0);

    // Sphere r = 0.6: closed, outward-facing, genus 0, accurate volume/area.
    auto sph = [](float x, float y, float z) { return shapes::sphere(x, y, z, 0.6f); };
    MeshStats s = mesh_of(sph, 96);
    check(s.watertight && s.oriented, "sphere is watertight and consistently oriented");
    check(s.components == 1 && s.genus == 0, "sphere: 1 shell, genus 0");
    check(s.volume > 0, "sphere normals point outward (positive volume)");
    check(rel_err(s.volume, 4.0 / 3 * pi * 0.216) < 0.01, "sphere volume within 1%");
    check(rel_err(s.area, 4 * pi * 0.36) < 0.02, "sphere area within 2%");

    // Torus R = 0.5, r = 0.2: one handle.
    auto tor = [](float x, float y, float z) { return shapes::torus(x, y, z, 0.5f, 0.2f); };
    s = mesh_of(tor, 96);
    check(s.watertight && s.oriented, "torus is watertight and consistently oriented");
    check(s.components == 1 && s.genus == 1, "torus: 1 shell, genus 1");
    check(rel_err(s.volume, 2 * pi * pi * 0.5 * 0.04) < 0.02, "torus volume within 2%");

    // Two separate spheres: two shells.
    auto two = [](float x, float y, float z) {
        return std::min(shapes::sphere(x - 0.5f, y, z, 0.3f), shapes::sphere(x + 0.5f, y, z, 0.3f));
    };
    s = mesh_of(two, 64);
    check(s.watertight && s.components == 2 && s.genus == 0, "two spheres: 2 closed shells");

    // Smooth unions (snowman) and very coarse grids stay closed. (Thin parts
    // can vanish between samples on coarse grids, so the coarse case uses a sphere.)
    s = mesh_of(shapes::snowman, 128);
    check(s.watertight && s.oriented && s.components == 1 && s.genus == 0, "snowman: closed, genus 0");
    for (int n : {3, 5, 8}) {
        s = mesh_of(sph, n);
        check(s.watertight && s.oriented, "coarse sphere n=" + std::to_string(n) + " stays watertight");
    }

    // Random fields hit all 256 cube cases: every non-empty mesh must be closed
    // and consistently oriented. This is what catches a bad table entry.
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    bool seen[256] = {};
    int meshes = 0, broken = 0;
    for (int trial = 0; trial < 20000; ++trial) {
        Grid g{3, -1.f, 1.f, std::vector<float>(64, 1.f)};  // outside on the border
        for (int k = 1; k < 3; ++k)
            for (int j = 1; j < 3; ++j)
                for (int i = 1; i < 3; ++i) g.values[(k * 4 + j) * 4 + i] = u(rng) + 1e-4f;
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i) {
                    int mask = 0;
                    for (int c = 0; c < 8; ++c)
                        if (g.at(i + (c == 1 || c == 2 || c == 5 || c == 6), j + (c == 2 || c == 3 || c == 6 || c == 7),
                                 k + (c >= 4)) < 0)
                            mask |= 1 << c;
                    seen[mask] = true;
                }
        const Mesh mesh = marching_cubes(g);
        if (mesh.triangles.empty()) continue;
        ++meshes;
        const MeshStats r = analyze(mesh);
        if (!(r.watertight && r.oriented)) ++broken;
    }
    int cases = 0;
    for (bool b : seen) cases += b;
    check(cases == 256, "random fields exercise all 256 cube cases (" + std::to_string(cases) + ")");
    check(broken == 0, std::to_string(meshes) + " random meshes, " + std::to_string(broken) + " not closed/oriented");

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "all passed", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
