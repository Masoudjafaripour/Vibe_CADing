#include "vibecad/printability.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace vibecad {
namespace {

constexpr double kFar = 1e20;  // "infinite" squared distance that stays finite in arithmetic

// Felzenszwalb & Huttenlocher (2012): exact 1D squared Euclidean distance
// transform of f in O(n), via the lower envelope of parabolas.
void edt_1d(const double* f, int n, double* d, int* v, double* z) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    auto meet = [&](int q, int p) {  // where the parabolas rooted at q and p intersect
        return ((f[q] + double(q) * q) - (f[p] + double(p) * p)) / (2.0 * (q - p));
    };
    // Only finite samples are parabolas; mixing kFar values would cancel catastrophically.
    int k = -1;
    for (int q = 0; q < n; ++q) {
        if (f[q] >= kFar) continue;
        if (k < 0) {
            k = 0, v[0] = q, z[0] = -inf, z[1] = inf;
            continue;
        }
        double s = meet(q, v[k]);
        while (s <= z[k]) s = meet(q, v[--k]);
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = inf;
    }
    if (k < 0) {  // no seed on this line
        std::fill(d, d + n, kFar);
        return;
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < q) ++k;
        const double dq = q - v[k];
        d[q] = dq * dq + f[v[k]];
    }
}

// Exact 3D squared Euclidean distance transform (in cells) on an m^3 grid:
// `dist` holds 0 at seed samples and kFar elsewhere on input.
void edt_3d(std::vector<double>& dist, int m) {

    // Separable: one 1D pass per axis. stride = distance between neighbours along that axis.
    const size_t strides[3] = {1, size_t(m), size_t(m) * m};
    for (int axis = 0; axis < 3; ++axis) {
        const size_t stride = strides[axis];
        const size_t outer = strides[(axis + 1) % 3], inner = strides[(axis + 2) % 3];
#pragma omp parallel
        {
            std::vector<double> f(m), d(m), z(m + 1);
            std::vector<int> v(m);
#pragma omp for collapse(2) schedule(static)
            for (int a = 0; a < m; ++a)
                for (int b = 0; b < m; ++b) {
                    const size_t base = a * outer + b * inner;
                    for (int q = 0; q < m; ++q) f[q] = dist[base + q * stride];
                    edt_1d(f.data(), m, d.data(), v.data(), z.data());
                    for (int q = 0; q < m; ++q) dist[base + q * stride] = d[q];
                }
        }
    }
}

// Squared distance (in cells) from every inside sample to the nearest outside sample.
std::vector<double> distance_to_outside(const Grid& g, float iso) {
    std::vector<double> dist(g.values.size());
    for (size_t i = 0; i < dist.size(); ++i) dist[i] = g.values[i] < iso ? kFar : 0.0;
    edt_3d(dist, g.n + 1);
    return dist;
}

Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

uint32_t find(std::vector<uint32_t>& parent, uint32_t x) {
    while (parent[x] != x) x = parent[x] = parent[parent[x]];
    return x;
}

std::string fmt(const char* format, double a, double b = 0, double c = 0, double d = 0) {
    char buf[256];
    std::snprintf(buf, sizeof buf, format, a, b, c, d);
    return buf;
}

}  // namespace

ThicknessReport wall_thickness(const Grid& g, float min_wall, float iso) {
    ThicknessReport r;
    const int m = g.n + 1;
    const float h = g.step();
    const std::vector<double> dist = distance_to_outside(g, iso);
    auto at = [&](int i, int j, int k) { return dist[(size_t(k) * m + j) * m + i]; };

    // Medial ridge: inside samples at least as deep as all 26 neighbours. There the
    // distance to the surface is half the local wall thickness.
    std::vector<float> thick;
    double cx = 0, cy = 0, cz = 0;
    float best = 1e30f;
    for (int k = 1; k < m - 1; ++k)
        for (int j = 1; j < m - 1; ++j)
            for (int i = 1; i < m - 1; ++i) {
                const double d = at(i, j, k);
                if (d <= 0) continue;
                bool ridge = true;
                for (int dk = -1; dk <= 1 && ridge; ++dk)
                    for (int dj = -1; dj <= 1 && ridge; ++dj)
                        for (int di = -1; di <= 1; ++di)
                            if (at(i + di, j + dj, k + dk) > d) {
                                ridge = false;
                                break;
                            }
                if (!ridge) continue;
                // Nearest outside sample is ~half a cell beyond the surface: thickness = (2 sqrt(d) - 1) cells.
                const float t = float(2.0 * std::sqrt(d) - 1.0) * h;
                thick.push_back(t);
                const Vec3 p{g.lo + i * h, g.lo + j * h, g.lo + k * h};
                if (t < best) best = t, r.thinnest = p;
                if (t < min_wall) {
                    ++r.thin_voxels;
                    cx += p[0], cy += p[1], cz += p[2];
                }
            }

    r.ridge_voxels = long(thick.size());
    if (thick.empty()) return r;
    r.min_thickness = best;
    r.thin_fraction = double(r.thin_voxels) / r.ridge_voxels;
    (void)cx, (void)cy, (void)cz;

    // Opening: erode by the wall radius, dilate back; whatever is lost is too thin.
    // The surface sits ~half a cell before the nearest outside sample, hence the 0.5s.
    const double radius = 0.5 * min_wall / h;
    std::vector<double> to_core(dist.size());
    for (size_t i = 0; i < dist.size(); ++i)
        to_core[i] = (dist[i] > 0 && std::sqrt(dist[i]) - 0.5 >= radius) ? 0.0 : kFar;
    edt_3d(to_core, m);
    // Dilating back by r*sqrt(3) (not r) keeps the 90-degree edges and corners of thick
    // parts, which a plain opening would shave off; thin fins and sharp tips stay thin.
    const double reach = (radius * std::sqrt(3.0) + 0.5) * (radius * std::sqrt(3.0) + 0.5);
    double tx = 0, ty = 0, tz = 0;
    for (int k = 0; k < m; ++k)
        for (int j = 0; j < m; ++j)
            for (int i = 0; i < m; ++i) {
                const size_t idx = (size_t(k) * m + j) * m + i;
                if (dist[idx] <= 0) continue;
                ++r.solid_voxels;
                if (to_core[idx] > reach) {
                    ++r.thin_volume_voxels;
                    tx += g.lo + i * h, ty += g.lo + j * h, tz += g.lo + k * h;
                }
            }
    r.thin_volume_fraction = double(r.thin_volume_voxels) / std::max(1L, r.solid_voxels);
    if (r.thin_volume_voxels)
        r.thin_centroid = {float(tx / r.thin_volume_voxels), float(ty / r.thin_volume_voxels),
                           float(tz / r.thin_volume_voxels)};
    auto nth = [&](double q) {
        auto it = thick.begin() + size_t(q * (thick.size() - 1));
        std::nth_element(thick.begin(), it, thick.end());
        return *it;
    };
    r.p05_thickness = nth(0.05);
    r.median_thickness = nth(0.5);
    return r;
}

PrintReport check_print(const Mesh& mesh, const PrintSettings& s) {
    PrintReport r;
    const MeshStats stats = analyze(mesh);
    r.watertight = stats.watertight;
    r.shells = stats.components;
    r.volume_mm3 = stats.volume;
    r.area_mm2 = stats.area;
    if (mesh.vertices.empty()) return r;

    Vec3 lo = mesh.vertices[0], hi = mesh.vertices[0];
    for (const Vec3& v : mesh.vertices)
        for (int d = 0; d < 3; ++d) lo[d] = std::min(lo[d], v[d]), hi[d] = std::max(hi[d], v[d]);
    r.size_mm = sub(hi, lo);

    // Shells: lowest point of each connected part.
    std::vector<uint32_t> parent(mesh.vertices.size());
    std::iota(parent.begin(), parent.end(), 0u);
    for (const auto& t : mesh.triangles)
        parent[find(parent, t[0])] = find(parent, t[1]), parent[find(parent, t[1])] = find(parent, t[2]);
    std::unordered_map<uint32_t, float> shell_min_z;
    std::unordered_map<uint32_t, Vec3> shell_low_point;
    for (uint32_t v = 0; v < mesh.vertices.size(); ++v) {
        const uint32_t root = find(parent, v);
        auto [it, fresh] = shell_min_z.try_emplace(root, mesh.vertices[v][2]);
        if (fresh || mesh.vertices[v][2] < it->second) it->second = mesh.vertices[v][2], shell_low_point[root] = mesh.vertices[v];
    }

    // Faces: overhangs (need support) and flat bed contact.
    const double down_limit = std::sin(s.overhang_deg * std::acos(-1.0) / 180.0);
    Vec3 overhang_centre{};
    for (const auto& t : mesh.triangles) {
        const Vec3 &a = mesh.vertices[t[0]], &b = mesh.vertices[t[1]], &c = mesh.vertices[t[2]];
        const Vec3 n = cross(sub(b, a), sub(c, a));
        const double len = std::sqrt(double(n[0]) * n[0] + double(n[1]) * n[1] + double(n[2]) * n[2]);
        if (len == 0) continue;
        const double area = 0.5 * len, down = -n[2] / len;  // cosine between the normal and -Z
        const float zmax = std::max({a[2], b[2], c[2]}), zmin = std::min({a[2], b[2], c[2]});
        if (zmax <= s.plate_eps && down > 0.9) {
            r.contact_area_mm2 += area;
        } else if (zmin > s.plate_eps && down > down_limit) {
            r.overhang_area_mm2 += area;
            for (int d = 0; d < 3; ++d) overhang_centre[d] += float(area * (a[d] + b[d] + c[d]) / 3.0);
        }
    }
    r.overhang_fraction = r.area_mm2 > 0 ? r.overhang_area_mm2 / r.area_mm2 : 0;

    if (!r.watertight)
        r.issues.push_back({"not_watertight", "critical", "The mesh has holes or flipped faces, so a slicer cannot fill it.", {}, 0});

    for (const auto& [root, z] : shell_min_z)
        if (z > s.plate_eps) {
            ++r.floating_shells;
            r.issues.push_back({"floating", "critical",
                                fmt("A separate part floats %.1f mm above the build plate; it would print in mid-air.", z),
                                shell_low_point[root], z});
        }

    if (r.overhang_area_mm2 > 0.5 * std::max(1.0, r.area_mm2 * 0.001)) {
        for (float& v : overhang_centre) v = float(v / r.overhang_area_mm2);
        r.issues.push_back({"overhang", "warning",
                            fmt("%.0f mm^2 (%.0f%% of the surface) overhangs more than %.0f degrees and needs support.",
                                r.overhang_area_mm2, 100 * r.overhang_fraction, s.overhang_deg),
                            overhang_centre, r.overhang_fraction});
    }

    if (r.contact_area_mm2 < std::max(s.min_contact, s.min_contact_fraction * r.size_mm[0] * r.size_mm[1]))
        r.issues.push_back({"small_contact", "warning",
                            fmt("Only %.0f mm^2 (%.1f%% of its footprint) rests flat on the build plate; it may tip "
                                "over or detach.",
                                r.contact_area_mm2, 100 * r.contact_area_mm2 / std::max(1.0, double(r.size_mm[0]) * r.size_mm[1])),
                            {}, r.contact_area_mm2});

    if (r.size_mm[0] > s.bed_x || r.size_mm[1] > s.bed_y || r.size_mm[2] > s.bed_z)
        r.issues.push_back({"too_big", "critical",
                            fmt("The part is %.0f x %.0f x %.0f mm; the printer holds %.0f mm per side at most.",
                                r.size_mm[0], r.size_mm[1], r.size_mm[2], std::min({s.bed_x, s.bed_y, s.bed_z})),
                            {}, 0});
    return r;
}

Mesh weld(const std::vector<Vec3>& soup) {
    struct Hash {
        size_t operator()(const Vec3& v) const {
            uint32_t b[3];
            std::memcpy(b, v.data(), 12);
            return (size_t(b[0]) * 73856093u) ^ (size_t(b[1]) * 19349663u) ^ (size_t(b[2]) * 83492791u);
        }
    };
    Mesh mesh;
    std::unordered_map<Vec3, uint32_t, Hash> index;
    index.reserve(soup.size() / 2);
    std::array<uint32_t, 3> tri{};
    for (size_t i = 0; i < soup.size(); ++i) {
        auto [it, fresh] = index.try_emplace(soup[i], uint32_t(mesh.vertices.size()));
        if (fresh) mesh.vertices.push_back(soup[i]);
        tri[i % 3] = it->second;
        if (i % 3 == 2 && tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2]) mesh.triangles.push_back(tri);
    }
    return mesh;
}

}  // namespace vibecad
