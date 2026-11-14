// Simplification keeps shape and topology; smoothing keeps volume; GLB is well formed.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "vibecad/mesh_ops.hpp"
#include "vibecad/shapes.hpp"

using namespace vibecad;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? "pass" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
double rel(double a, double b) { return std::abs(a - b) / std::abs(b); }

// Largest |sdf| over the vertices: how far the simplified surface strays from the true one.
template <class SDF>
double max_deviation(const Mesh& m, SDF&& sdf) {
    double worst = 0;
    for (const Vec3& v : m.vertices) worst = std::max(worst, double(std::abs(sdf(v[0], v[1], v[2]))));
    return worst;
}
}  // namespace

int main() {
    const double pi = std::acos(-1.0);
    auto sph = [](float x, float y, float z) { return shapes::sphere(x, y, z, 0.6f); };
    auto tor = [](float x, float y, float z) { return shapes::torus(x, y, z, 0.5f, 0.2f); };
    auto two = [](float x, float y, float z) {
        return std::min(shapes::sphere(x - 0.5f, y, z, 0.3f), shapes::sphere(x + 0.5f, y, z, 0.3f));
    };

    // Simplify a sphere 10x: still closed, genus 0, volume and surface within ~1% of the true sphere.
    Mesh m = marching_cubes(sample(sph, 128));
    const size_t before = m.triangles.size();
    SimplifyReport r = simplify(m, before / 10);
    MeshStats s = analyze(m);
    check(r.triangles_after <= before / 10 + 2, "sphere: " + std::to_string(before) + " -> " +
                                                     std::to_string(r.triangles_after) + " triangles");
    check(s.watertight && s.oriented && s.components == 1 && s.genus == 0, "sphere: still watertight, genus 0");
    check(rel(s.volume, 4.0 / 3 * pi * 0.216) < 0.01, "sphere: volume within 1% of 4/3 pi r^3");
    check(max_deviation(m, sph) < 0.006, "sphere: vertices within 1% of the radius from the true surface");

    // Topology survives aggressive simplification: torus keeps its hole, two spheres stay two.
    m = marching_cubes(sample(tor, 128));
    simplify(m, 600);
    s = analyze(m);
    check(s.watertight && s.genus == 1 && s.components == 1, "torus at 600 triangles: genus 1 kept");
    check(rel(s.volume, 2 * pi * pi * 0.5 * 0.04) < 0.03, "torus: volume within 3%");
    m = marching_cubes(sample(two, 96));
    simplify(m, 300);
    s = analyze(m);
    check(s.watertight && s.components == 2 && s.genus == 0, "two spheres at 300 triangles: 2 closed shells kept");

    // Taubin smoothing is a denoiser. Marching cubes on an exact SDF is already accurate
    // (smoothing it makes normals slightly worse), so test it on a noisy mesh instead.
    m = marching_cubes(sample(sph, 48));
    std::mt19937 rng(3);
    std::normal_distribution<float> noise(0.f, 0.004f);
    for (Vec3& v : m.vertices)
        for (float& x : v) x += noise(rng);
    auto normal_error_deg = [](const Mesh& mm) {
        double sum = 0;
        for (const auto& t : mm.triangles) {
            const Vec3 &a = mm.vertices[t[0]], &b = mm.vertices[t[1]], &c = mm.vertices[t[2]];
            const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, w[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
            const double n[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
            const double g[3] = {(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0};
            const double cosang = (n[0] * g[0] + n[1] * g[1] + n[2] * g[2]) /
                                  (std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) *
                                   std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]));
            sum += std::acos(std::min(1.0, cosang)) * 180 / 3.14159265358979;
        }
        return sum / mm.triangles.size();
    };
    const double vol0 = analyze(m).volume, err0 = normal_error_deg(m);
    smooth_taubin(m, 10);
    const double err1 = normal_error_deg(m);
    check(err1 < 0.5 * err0, "Taubin on a noisy sphere: face-normal error " + std::to_string(err0) + " -> " + std::to_string(err1) + " deg");
    check(rel(analyze(m).volume, vol0) < 0.005, "Taubin: volume change < 0.5% (no shrinking)");
    check(max_deviation(m, sph) < 0.01, "Taubin: vertices end within 1.7% of the radius from the true surface");

    // Normals on a sphere point outward.
    const std::vector<Vec3> n = vertex_normals(m);
    double worst = 1;
    for (size_t i = 0; i < n.size(); ++i) {
        const Vec3& v = m.vertices[i];
        const double len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        worst = std::min(worst, (n[i][0] * v[0] + n[i][1] * v[1] + n[i][2] * v[2]) / len);
    }
    check(worst > 0.95, "vertex normals point outward (min cos " + std::to_string(worst) + ")");

    // GLB: header, chunk types and lengths are consistent.
    std::vector<Vec3> colors(m.vertices.size(), Vec3{1.f, 0.5f, 0.f});
    write_glb(m, "test_mesh_ops.glb", colors, "sphere");
    std::ifstream f("test_mesh_ops.glb", std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::remove("test_mesh_ops.glb");
    uint32_t h[5];
    std::memcpy(h, data.data(), 20);
    const std::string json(data.data() + 20, h[3]);
    uint32_t bin_len, bin_type;
    std::memcpy(&bin_len, data.data() + 20 + h[3], 4);
    std::memcpy(&bin_type, data.data() + 24 + h[3], 4);
    check(h[0] == 0x46546C67 && h[1] == 2 && h[2] == data.size() && h[4] == 0x4E4F534A && bin_type == 0x004E4942,
          "GLB header and chunk types");
    check(json.find("\"COLOR_0\":2") != std::string::npos && json.find("\"count\":" + std::to_string(m.triangles.size() * 3)) != std::string::npos &&
              20 + h[3] + 8 + bin_len == data.size(),
          "GLB JSON lists colours and the right index count; lengths add up");

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "all passed", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
