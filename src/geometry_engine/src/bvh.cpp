// BVH writer. Channel order "Xrotation Yrotation Zrotation" means R = Rx * Ry * Rz,
// the same convention as skeleton.euler_to_matrix, so angles are written as-is (in degrees).
#include <cmath>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>

#include "vibecad/motion.hpp"
#include "vibecad/motion_models.hpp"

namespace vibecad::motion {
namespace {

void emit_bvh(std::ostream& out, const Skeleton& skel, const double* root, const double* angles, int frames, double fps) {
    const int J = skel.size();
    std::vector<std::vector<int>> children(J);
    for (int j = 0; j < J; ++j)
        if (skel.parents[j] >= 0) children[skel.parents[j]].push_back(j);

    out.setf(std::ios::fixed);
    out.precision(6);

    std::vector<int> order;  // depth-first order = order of the motion channels
    std::function<void(int, int)> emit = [&](int j, int depth) {
        const std::string pad(depth * 2, ' ');
        const Vec3& o = skel.offsets[j];
        out << pad << (skel.parents[j] < 0 ? "ROOT " : "JOINT ") << skel.names[j] << "\n" << pad << "{\n";
        out << pad << "  OFFSET " << o[0] << " " << o[1] << " " << o[2] << "\n";
        out << pad << "  CHANNELS " << (skel.parents[j] < 0 ? "6 Xposition Yposition Zposition " : "3 ")
            << "Xrotation Yrotation Zrotation\n";
        order.push_back(j);
        for (int c : children[j]) emit(c, depth + 1);
        if (children[j].empty()) {  // leaf: a short end bone continuing the last segment
            const double len = std::sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]);
            const Vec3 e = len > 0 ? Vec3{o[0] * 0.3, o[1] * 0.3, o[2] * 0.3} : Vec3{0, 0.05, 0};
            out << pad << "  End Site\n" << pad << "  {\n"
                << pad << "    OFFSET " << e[0] << " " << e[1] << " " << e[2] << "\n" << pad << "  }\n";
        }
        out << pad << "}\n";
    };

    out << "HIERARCHY\n";
    for (int j = 0; j < J; ++j)
        if (skel.parents[j] < 0) emit(j, 0);

    constexpr double kDeg = 180.0 / 3.14159265358979323846;
    out << "MOTION\nFrames: " << frames << "\nFrame Time: " << 1.0 / fps << "\n";
    for (int t = 0; t < frames; ++t) {
        out << root[t * 3] << " " << root[t * 3 + 1] << " " << root[t * 3 + 2];
        for (int j : order) {
            const double* a = angles + (size_t(t) * J + j) * 3;
            out << " " << a[0] * kDeg << " " << a[1] * kDeg << " " << a[2] * kDeg;
        }
        out << "\n";
    }
}

}  // namespace

void write_bvh(const std::string& path, const Skeleton& skel, const double* root, const double* angles, int frames,
               double fps) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open " + path);
    emit_bvh(out, skel, root, angles, frames, fps);
}

std::string bvh_text(const Skeleton& skel, const double* root, const double* angles, int frames, double fps) {
    std::ostringstream out;
    emit_bvh(out, skel, root, angles, frames, fps);
    return out.str();
}

}  // namespace vibecad::motion
