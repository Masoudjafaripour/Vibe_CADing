// QEM edge-collapse simplification with topology-preserving guards.
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "vibecad/mesh_ops.hpp"

namespace vibecad {
namespace {

// Symmetric 4x4 quadric stored as its 10 unique entries.
struct Quadric {
    double a[10] = {};

    static Quadric plane(double nx, double ny, double nz, double d, double w) {
        Quadric q;
        const double v[4] = {nx, ny, nz, d};
        int k = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = i; j < 4; ++j) q.a[k++] = w * v[i] * v[j];
        return q;
    }
    Quadric& operator+=(const Quadric& o) {
        for (int i = 0; i < 10; ++i) a[i] += o.a[i];
        return *this;
    }
    // v^T Q v with v = (x, y, z, 1)
    double error(double x, double y, double z) const {
        return a[0] * x * x + 2 * a[1] * x * y + 2 * a[2] * x * z + 2 * a[3] * x + a[4] * y * y + 2 * a[5] * y * z +
               2 * a[6] * y + a[7] * z * z + 2 * a[8] * z + a[9];
    }
    // Position minimising the error, if the 3x3 system is well conditioned.
    bool optimum(double& x, double& y, double& z) const {
        const double m00 = a[0], m01 = a[1], m02 = a[2], m11 = a[4], m12 = a[5], m22 = a[7];
        const double b0 = -a[3], b1 = -a[6], b2 = -a[8];
        const double c00 = m11 * m22 - m12 * m12, c01 = m02 * m12 - m01 * m22, c02 = m01 * m12 - m02 * m11;
        const double det = m00 * c00 + m01 * c01 + m02 * c02;
        const double scale = std::abs(m00) + std::abs(m11) + std::abs(m22);
        if (std::abs(det) < 1e-12 * scale * scale * scale) return false;
        const double c11 = m00 * m22 - m02 * m02, c12 = m01 * m02 - m00 * m12, c22 = m00 * m11 - m01 * m01;
        x = (c00 * b0 + c01 * b1 + c02 * b2) / det;
        y = (c01 * b0 + c11 * b1 + c12 * b2) / det;
        z = (c02 * b0 + c12 * b1 + c22 * b2) / det;
        return true;
    }
};

Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double dot(const Vec3& a, const Vec3& b) { return double(a[0]) * b[0] + double(a[1]) * b[1] + double(a[2]) * b[2]; }

struct Candidate {
    double cost;
    uint32_t u, v;
    uint32_t stamp_u, stamp_v;  // vertex versions when computed; stale entries are skipped
    Vec3 target;
    bool operator<(const Candidate& o) const { return cost > o.cost; }  // min-heap
};

}  // namespace

SimplifyReport simplify(Mesh& mesh, size_t target) {
    SimplifyReport rep;
    rep.triangles_before = mesh.triangles.size();
    const size_t nv = mesh.vertices.size(), nf = mesh.triangles.size();
    if (nf <= target || nf < 4) {
        rep.triangles_after = nf;
        return rep;
    }

    auto& V = mesh.vertices;
    auto& F = mesh.triangles;
    std::vector<char> face_alive(nf, 1), vert_alive(nv, 1);
    std::vector<std::vector<uint32_t>> vf(nv);  // vertex -> incident faces (may hold dead ones)
    std::vector<Quadric> Q(nv);
    std::vector<uint32_t> stamp(nv, 0);

    for (uint32_t f = 0; f < nf; ++f) {
        const Vec3 n = cross(sub(V[F[f][1]], V[F[f][0]]), sub(V[F[f][2]], V[F[f][0]]));
        const double len = std::sqrt(dot(n, n));
        for (uint32_t i : F[f]) vf[i].push_back(f);
        if (len == 0) continue;
        const double nx = n[0] / len, ny = n[1] / len, nz = n[2] / len;
        const double d = -(nx * V[F[f][0]][0] + ny * V[F[f][0]][1] + nz * V[F[f][0]][2]);
        const Quadric q = Quadric::plane(nx, ny, nz, d, 0.5 * len);  // area-weighted
        for (uint32_t i : F[f]) Q[i] += q;
    }

    auto ring = [&](uint32_t v, std::vector<uint32_t>& out) {  // live one-ring vertices
        out.clear();
        for (uint32_t f : vf[v])
            if (face_alive[f])
                for (uint32_t w : F[f])
                    if (w != v && std::find(out.begin(), out.end(), w) == out.end()) out.push_back(w);
    };

    auto evaluate = [&](uint32_t u, uint32_t v) {
        Quadric q = Q[u];
        q += Q[v];
        Candidate c{0, u, v, stamp[u], stamp[v], {}};
        double x, y, z;
        if (q.optimum(x, y, z)) {
            // keep the optimum only if it stays near the edge (guards against ill-conditioned flat regions)
            const Vec3 mid{(V[u][0] + V[v][0]) / 2, (V[u][1] + V[v][1]) / 2, (V[u][2] + V[v][2]) / 2};
            const Vec3 e = sub(V[u], V[v]);
            const double dx = x - mid[0], dy = y - mid[1], dz = z - mid[2];
            if (dx * dx + dy * dy + dz * dz <= dot(e, e)) {
                c.target = {float(x), float(y), float(z)};
                c.cost = q.error(x, y, z);
                return c;
            }
        }
        const Vec3 cand[3] = {V[u], V[v], {(V[u][0] + V[v][0]) / 2, (V[u][1] + V[v][1]) / 2, (V[u][2] + V[v][2]) / 2}};
        c.cost = 1e300;
        for (const Vec3& p : cand) {
            const double e = q.error(p[0], p[1], p[2]);
            if (e < c.cost) c.cost = e, c.target = p;
        }
        return c;
    };

    std::priority_queue<Candidate> heap;
    {
        std::unordered_set<uint64_t> seen;
        seen.reserve(nf * 2);
        for (uint32_t f = 0; f < nf; ++f)
            for (int e = 0; e < 3; ++e) {
                uint32_t a = F[f][e], b = F[f][(e + 1) % 3];
                if (a > b) std::swap(a, b);
                if (seen.insert(uint64_t(a) << 32 | b).second) heap.push(evaluate(a, b));
            }
    }

    size_t faces = nf;
    std::vector<uint32_t> ring_u, ring_v, shared;
    while (faces > target && !heap.empty()) {
        const Candidate c = heap.top();
        heap.pop();
        const uint32_t u = c.u, v = c.v;
        if (!vert_alive[u] || !vert_alive[v] || stamp[u] != c.stamp_u || stamp[v] != c.stamp_v) continue;

        // Faces on the edge; a manifold interior edge has exactly two.
        int on_edge = 0;
        for (uint32_t f : vf[u])
            if (face_alive[f] && (F[f][0] == v || F[f][1] == v || F[f][2] == v)) ++on_edge;
        if (on_edge != 2) continue;  // boundary or non-manifold edge: leave it

        // Link condition: the one-rings of u and v may share only the two opposite vertices.
        ring(u, ring_u);
        ring(v, ring_v);
        shared.clear();
        for (uint32_t w : ring_u)
            if (std::find(ring_v.begin(), ring_v.end(), w) != ring_v.end()) shared.push_back(w);
        if (shared.size() != 2 || ring_u.size() <= 3 || ring_v.size() <= 3) {
            ++rep.rejected_topology;
            continue;
        }

        // No surviving face may flip or collapse to a sliver.
        bool ok = true;
        for (uint32_t w : {u, v}) {
            for (uint32_t f : vf[w]) {
                if (!face_alive[f]) continue;
                const auto& t = F[f];
                if ((t[0] == u || t[1] == u || t[2] == u) && (t[0] == v || t[1] == v || t[2] == v)) continue;
                Vec3 p[3] = {V[t[0]], V[t[1]], V[t[2]]};
                const Vec3 before = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                for (int k = 0; k < 3; ++k)
                    if (t[k] == w) p[k] = c.target;
                const Vec3 after = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                const double lb = std::sqrt(dot(before, before)), la = std::sqrt(dot(after, after));
                if (la < 1e-12 || dot(before, after) < 0.2 * lb * la) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }
        if (!ok) {
            ++rep.rejected_flip;
            continue;
        }

        // Collapse v into u.
        V[u] = c.target;
        Q[u] += Q[v];
        vert_alive[v] = 0;
        for (uint32_t f : vf[v]) {
            if (!face_alive[f]) continue;
            auto& t = F[f];
            if (t[0] == u || t[1] == u || t[2] == u) {
                face_alive[f] = 0;
                --faces;
            } else {
                for (uint32_t& w : t)
                    if (w == v) w = u;
                vf[u].push_back(f);
            }
        }
        vf[v].clear();
        auto& list = vf[u];  // drop dead faces from u's list
        list.erase(std::remove_if(list.begin(), list.end(), [&](uint32_t f) { return !face_alive[f]; }), list.end());

        ++stamp[u];
        ring(u, ring_u);
        for (uint32_t w : ring_u) {
            ++stamp[w];  // its other edges' costs are unchanged, but their stamps moved; re-push all edges around w
        }
        for (uint32_t w : ring_u) {
            std::vector<uint32_t> rw;
            ring(w, rw);
            for (uint32_t x : rw) heap.push(evaluate(std::min(w, x), std::max(w, x)));
        }
    }

    // Compact.
    std::vector<uint32_t> remap(nv, UINT32_MAX);
    Mesh out;
    for (uint32_t f = 0; f < nf; ++f) {
        if (!face_alive[f]) continue;
        std::array<uint32_t, 3> t;
        for (int k = 0; k < 3; ++k) {
            uint32_t& r = remap[F[f][k]];
            if (r == UINT32_MAX) r = uint32_t(out.vertices.size()), out.vertices.push_back(V[F[f][k]]);
            t[k] = r;
        }
        out.triangles.push_back(t);
    }
    mesh = std::move(out);
    rep.triangles_after = mesh.triangles.size();
    return rep;
}

}  // namespace vibecad
