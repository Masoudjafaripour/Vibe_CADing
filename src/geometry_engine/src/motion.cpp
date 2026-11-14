#include "vibecad/motion.hpp"

#include <algorithm>
#include <cmath>

namespace vibecad::motion {
namespace {

using Mat3 = std::array<std::array<double, 3>, 3>;

Mat3 mul(const Mat3& a, const Mat3& b) {
    Mat3 r{};
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j) r[i][j] += a[i][k] * b[k][j];
    return r;
}

Vec3 mul(const Mat3& a, const Vec3& v) {
    return {a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2], a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
            a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2]};
}

// Same convention as skeleton.euler_to_matrix: R = Rx @ Ry @ Rz.
Mat3 euler_xyz(const double* a) {
    const double cx = std::cos(a[0]), sx = std::sin(a[0]), cy = std::cos(a[1]), sy = std::sin(a[1]),
                 cz = std::cos(a[2]), sz = std::sin(a[2]);
    const Mat3 rx{{{1, 0, 0}, {0, cx, -sx}, {0, sx, cx}}};
    const Mat3 ry{{{cy, 0, sy}, {0, 1, 0}, {-sy, 0, cy}}};
    const Mat3 rz{{{cz, -sz, 0}, {sz, cz, 0}, {0, 0, 1}}};
    return mul(mul(rx, ry), rz);
}

Vec3 sub(const double* a, const double* b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double norm(const Vec3& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
double norm_xz(double dx, double dz) { return std::sqrt(dx * dx + dz * dz); }

}  // namespace

void forward_kinematics(const Skeleton& skel, const double* root, const double* angles, int frames, double* pos) {
    const int J = skel.size();
    std::vector<Mat3> glob(J);
    for (int t = 0; t < frames; ++t) {
        for (int j = 0; j < J; ++j) {
            const Mat3 local = euler_xyz(angles + (size_t(t) * J + j) * 3);
            double* p = pos + (size_t(t) * J + j) * 3;
            const int parent = skel.parents[j];
            if (parent < 0) {
                std::copy(root + size_t(t) * 3, root + size_t(t) * 3 + 3, p);
                glob[j] = local;
            } else {
                const double* pp = pos + (size_t(t) * J + parent) * 3;
                const Vec3 o = mul(glob[parent], skel.offsets[j]);
                for (int d = 0; d < 3; ++d) p[d] = pp[d] + o[d];
                glob[j] = mul(glob[parent], local);
            }
        }
    }
}

std::vector<int> stance_toes(const Clip& pos, int l_toe, int r_toe, double fps, double contact_height,
                             double max_speed) {
    const int T = pos.frames;
    std::vector<int> stance(T, -1);
    if (T < 2) return stance;
    const int toes[2] = {l_toe, r_toe};
    for (int t = 0; t < T; ++t) {
        // np.gradient: one-sided at the ends, central inside
        const int a = t == 0 ? 0 : t - 1, b = t == T - 1 ? T - 1 : t + 1;
        const double scale = (b - a == 2 ? 0.5 : 1.0) * fps;
        double speed[2];
        for (int k = 0; k < 2; ++k) {
            const double* pa = pos.at(a, toes[k]);
            const double* pb = pos.at(b, toes[k]);
            speed[k] = norm_xz(pb[0] - pa[0], pb[2] - pa[2]) * scale;
        }
        const int slow = speed[1] < speed[0] ? 1 : 0;  // np.argmin: first on ties
        if (speed[slow] < max_speed && pos.at(t, toes[slow])[1] < contact_height) stance[t] = toes[slow];
    }
    return stance;
}

void lock_feet(const Clip& pos, const Leg& left, const Leg& right, double fps, double contact_height,
               double max_drift, double* out) {
    const int T = pos.frames, J = pos.joints;
    std::copy(pos.data, pos.data + size_t(T) * J * 3, out);
    const std::vector<int> stance = stance_toes(pos, left.toe, right.toe, fps, contact_height, 1.0);
    auto o = [&](int t, int j) { return out + (size_t(t) * J + j) * 3; };

    for (const Leg& leg : {left, right}) {
        const double thigh = norm(sub(pos.at(0, leg.knee), pos.at(0, leg.hip)));
        const double shin = norm(sub(pos.at(0, leg.ankle), pos.at(0, leg.knee)));
        bool locked = false;
        Vec3 anchor{}, foot{};
        for (int t = 0; t < T; ++t) {
            if (stance[t] != leg.toe) {
                locked = false;
                continue;
            }
            const double* toe = pos.at(t, leg.toe);
            if (!locked) {  // contact begins: pin the toe on the floor where it is
                anchor = {toe[0], 0.0, toe[2]};
                foot = sub(pos.at(t, leg.ankle), toe);
                locked = true;
            }
            const double* hip = o(t, leg.hip);
            const Vec3 d{anchor[0] + foot[0] - hip[0], anchor[1] + foot[1] - hip[1], anchor[2] + foot[2] - hip[2]};
            const double length = norm(d);
            const double drift = norm({toe[0] - anchor[0], toe[1] - anchor[1], toe[2] - anchor[2]});
            if (length > thigh + shin - 1e-3 || drift > max_drift) {  // can't reach, or foot moved on
                locked = false;
                continue;
            }
            // Two-bone IK: knee on the circle where thigh and shin spheres meet,
            // on the side the original knee bends to.
            const double along = (thigh * thigh - shin * shin + length * length) / (2 * length);
            const Vec3 axis{d[0] / length, d[1] / length, d[2] / length};
            Vec3 pole = sub(pos.at(t, leg.knee), pos.at(t, leg.hip));
            const double dp = pole[0] * axis[0] + pole[1] * axis[1] + pole[2] * axis[2];
            for (int k = 0; k < 3; ++k) pole[k] -= axis[k] * dp;
            const double pn = norm(pole) + 1e-9;
            const double h = std::sqrt(std::max(thigh * thigh - along * along, 0.0));
            double* knee = o(t, leg.knee);
            double* ankle = o(t, leg.ankle);
            double* toe_out = o(t, leg.toe);
            for (int k = 0; k < 3; ++k) {
                knee[k] = hip[k] + axis[k] * along + pole[k] / pn * h;
                ankle[k] = anchor[k] + foot[k];
                toe_out[k] = anchor[k];
            }
        }
    }
}

double foot_sliding(const Clip& pos, int l_toe, int r_toe, double fps, double contact_height) {
    double sum = 0;
    long count = 0;
    for (int t = 1; t < pos.frames; ++t)
        for (int toe : {l_toe, r_toe}) {
            const double* a = pos.at(t - 1, toe);
            const double* b = pos.at(t, toe);
            if (b[1] < contact_height) {
                sum += norm_xz(b[0] - a[0], b[2] - a[2]) * fps;
                ++count;
            }
        }
    return count ? sum / count : 0.0;
}

double stance_sliding(const Clip& pos, const Clip& reference, int l_toe, int r_toe, double fps) {
    const std::vector<int> stance = stance_toes(reference, l_toe, r_toe, fps, 0.03, 1.0);
    double sum = 0;
    long count = 0;
    for (int t = 1; t < pos.frames; ++t) {
        if (stance[t] < 0) continue;
        const double* a = pos.at(t - 1, stance[t]);
        const double* b = pos.at(t, stance[t]);
        sum += norm_xz(b[0] - a[0], b[2] - a[2]);
        ++count;
    }
    return count ? sum / count * fps : 0.0;
}

double jerk(const Clip& pos, double fps) {
    if (pos.frames < 4) return 0.0;
    const double f3 = fps * fps * fps;
    double sum = 0;
    for (int t = 0; t + 3 < pos.frames; ++t)
        for (int j = 0; j < pos.joints; ++j) {
            const double *p0 = pos.at(t, j), *p1 = pos.at(t + 1, j), *p2 = pos.at(t + 2, j), *p3 = pos.at(t + 3, j);
            Vec3 d;
            for (int k = 0; k < 3; ++k) d[k] = ((p3[k] - p2[k]) - 2 * (p2[k] - p1[k]) + (p1[k] - p0[k])) * f3;
            sum += norm(d);
        }
    return sum / (double(pos.frames - 3) * pos.joints);
}

}  // namespace vibecad::motion
