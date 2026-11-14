// Printability checks on shapes with known answers.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "vibecad/printability.hpp"

using namespace vibecad;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? "pass" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

float box(float x, float y, float z, float bx, float by, float bz) {
    const float qx = std::abs(x) - bx, qy = std::abs(y) - by, qz = std::abs(z) - bz;
    const float ox = std::max(qx, 0.f), oy = std::max(qy, 0.f), oz = std::max(qz, 0.f);
    return std::sqrt(ox * ox + oy * oy + oz * oz) + std::min(std::max(qx, std::max(qy, qz)), 0.f);
}

// Meshes a Z-up SDF over [-1, 1]^3 and scales it to millimetres (1 unit = 50 mm), resting on z = 0.
template <class SDF>
Mesh print_mesh(SDF&& sdf, int n = 96) {
    Mesh m = marching_cubes(sample(sdf, n));
    float zmin = 1e9f;
    for (auto& v : m.vertices) zmin = std::min(zmin, v[2]);
    for (auto& v : m.vertices) v = {v[0] * 50, v[1] * 50, (v[2] - zmin) * 50};
    return m;
}

bool has(const PrintReport& r, const std::string& kind) {
    return std::any_of(r.issues.begin(), r.issues.end(), [&](const Issue& i) { return i.kind == kind; });
}

}  // namespace

int main() {
    const double pi = std::acos(-1.0);

    // Wall thickness: a 0.3-thick slab, a sphere of diameter 1.0, a too-thin 0.06 slab.
    const float h = 2.f / 100;
    ThicknessReport t = wall_thickness(sample([](float x, float y, float z) { return box(x, y, z, 0.6f, 0.6f, 0.15f); }, 100), 0.1f);
    check(std::abs(t.median_thickness - 0.3f) <= 1.5f * h, "slab 0.30 thick measures " + std::to_string(t.median_thickness));
    check(t.thin_voxels == 0, "slab 0.30 has no walls under 0.10");

    t = wall_thickness(sample([](float x, float y, float z) { return std::sqrt(x * x + y * y + z * z) - 0.5f; }, 100), 0.1f);
    check(std::abs(t.median_thickness - 1.0f) <= 1.5f * h, "sphere of diameter 1.0 measures " + std::to_string(t.median_thickness));

    t = wall_thickness(sample([](float x, float y, float z) { return box(x, y, z, 0.6f, 0.6f, 0.03f); }, 100), 0.1f);
    check(t.thin_fraction > 0.95 && std::abs(t.thinnest[2]) < 2 * h, "slab 0.06 thick is flagged thin, located at z = 0");

    // Opening: a 0.04-thick fin on a thick block; only the fin is thin, by volume.
    auto fin = [](float x, float y, float z) {
        return std::min(box(x, y, z, 0.4f, 0.4f, 0.4f), box(x - 0.6f, y, z, 0.2f, 0.4f, 0.02f));
    };
    t = wall_thickness(sample(fin, 100), 0.1f);
    const double fin_share = (0.4 * 0.8 * 0.04) / (0.8 * 0.8 * 0.8 + 0.4 * 0.8 * 0.04);
    check(std::abs(t.thin_volume_fraction - fin_share) < 0.01 && t.thin_centroid[0] > 0.45f,
          "block + thin fin: thin volume " + std::to_string(t.thin_volume_fraction) + " ~ fin share " +
              std::to_string(fin_share) + ", located on the fin");
    t = wall_thickness(sample([](float x, float y, float z) { return box(x, y, z, 0.6f, 0.6f, 0.15f); }, 100), 0.1f);
    // (only grid-level noise at the 8 box vertices: < 0.1% of the volume)
    check(t.thin_volume_fraction < 0.001, "slab 0.30 thick: no thin volume at min wall 0.10");

    // A box standing on the plate: no overhang, large flat contact, nothing floating.
    PrintReport r = check_print(print_mesh([](float x, float y, float z) { return box(x, y, z, 0.4f, 0.4f, 0.4f); }));
    check(r.watertight && r.floating_shells == 0 && r.overhang_area_mm2 < 1, "box: watertight, no overhang, no floating parts");
    check(std::abs(r.contact_area_mm2 - 40.0 * 40) / (40.0 * 40) < 0.05, "box: bed contact = its 40 x 40 mm footprint");
    check(r.issues.empty(), "box: no issues");

    // Mushroom: stem r = 0.15 under a cap r = 0.6. The cap's underside ring overhangs.
    auto mushroom = [](float x, float y, float z) {
        const float rho = std::sqrt(x * x + y * y);
        const float stem = std::max(rho - 0.15f, std::abs(z + 0.4f) - 0.4f);
        const float cap = std::max(rho - 0.6f, std::abs(z - 0.1f) - 0.1f);
        return std::min(stem, cap);
    };
    r = check_print(print_mesh(mushroom));
    const double ring = pi * (30.0 * 30 - 7.5 * 7.5);  // mm^2 at 50 mm per unit
    check(has(r, "overhang") && std::abs(r.overhang_area_mm2 - ring) / ring < 0.08,
          "mushroom: cap underside overhang " + std::to_string(int(r.overhang_area_mm2)) + " ~ " + std::to_string(int(ring)) + " mm^2");

    // Two boxes, one raised above the plate: one floating part.
    auto two = [](float x, float y, float z) {
        return std::min(box(x + 0.4f, y, z + 0.6f, 0.2f, 0.2f, 0.2f), box(x - 0.4f, y, z - 0.2f, 0.2f, 0.2f, 0.2f));
    };
    r = check_print(print_mesh(two));
    check(r.shells == 2 && r.floating_shells == 1 && has(r, "floating"), "two boxes: the raised one floats");

    // A sphere touches the plate at a point: small-contact warning.
    r = check_print(print_mesh([](float x, float y, float z) { return std::sqrt(x * x + y * y + z * z) - 0.5f; }));
    check(has(r, "small_contact") && r.floating_shells == 0, "sphere: small bed contact warning");

    // Build volume: a 2 m part does not fit.
    Mesh big = print_mesh([](float x, float y, float z) { return box(x, y, z, 0.4f, 0.4f, 0.4f); });
    for (auto& v : big.vertices) v = {v[0] * 50, v[1] * 50, v[2] * 50};
    check(has(check_print(big), "too_big"), "a 2 m cube is too big for the printer");

    // STL round trip: write, read back, weld -> the same closed mesh.
    const Mesh m = print_mesh(mushroom, 48);
    write_stl(m, "test_print_roundtrip.stl");
    const Mesh back = read_stl("test_print_roundtrip.stl");
    std::remove("test_print_roundtrip.stl");
    check(back.triangles.size() == m.triangles.size() && back.vertices.size() == m.vertices.size() &&
              analyze(back).watertight,
          "STL write -> read -> weld gives the same watertight mesh");

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "all passed", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
