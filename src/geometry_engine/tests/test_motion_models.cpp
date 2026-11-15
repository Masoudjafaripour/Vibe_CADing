// The C++ text-to-motion models produce the motion the text asks for.
// (Exact agreement with PyTorch is checked in src/animation/test_motion_models.py.)
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include "vibecad/motion_models.hpp"

using namespace vibecad::motion;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? "pass" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
int joint(const Skeleton& s, const std::string& name) {
    for (int j = 0; j < s.size(); ++j)
        if (s.names[j] == name) return j;
    return -1;
}
struct Summary {
    double forward = 0;     // root travel along +z, m
    double wrist_over = -9; // max (right wrist y - head y), m
    double arms_over = -9;  // max over frames of min(left, right wrist y) - head y, m
};
Summary summarize(const MotionModel& m, const GeneratedClip& c) {
    const Skeleton& s = m.skeleton();
    std::vector<double> pos(size_t(c.frames) * s.size() * 3);
    forward_kinematics(s, c.root.data(), c.angles.data(), c.frames, pos.data());
    const int head = joint(s, "head"), lw = joint(s, "l_wrist"), rw = joint(s, "r_wrist");
    auto y = [&](int t, int j) { return pos[(size_t(t) * s.size() + j) * 3 + 1]; };
    Summary r;
    r.forward = c.root[(c.frames - 1) * 3 + 2] - c.root[2];
    for (int t = 0; t < c.frames; ++t) {
        r.wrist_over = std::max(r.wrist_over, y(t, rw) - y(t, head));
        r.arms_over = std::max(r.arms_over, std::min(y(t, lw), y(t, rw)) - y(t, head));
    }
    return r;
}
}  // namespace

int main() {
    const std::string dir = VIBECAD_MOTION_WEIGHTS;
    if (!std::ifstream(dir + "/ar.vcw")) {
        std::printf("[skip] no weights in %s (run src/animation/export_weights.py)\n", dir.c_str());
        return 0;
    }
    for (const char* method : {"ar", "diffusion"}) {
        const auto m = MotionModel::load_file(dir + "/" + method + ".vcw");
        check(m->method() == method && m->skeleton().size() == 18 && m->frames() == 90, std::string(method) + ": loads (18 joints, 90 frames)");
        const GeneratedClip walk = m->generate("walk forward", 1), wave = m->generate("wave hello", 1), arms = m->generate("raise both arms", 1);
        const Summary sw = summarize(*m, walk), sv = summarize(*m, wave), sa = summarize(*m, arms);
        char buf[200];
        std::snprintf(buf, sizeof buf, "%s: \"walk forward\" moves forward (%.2f m in 3 s, %.0f ms)", method, sw.forward, walk.ms);
        check(sw.forward > 1.0, buf);
        std::snprintf(buf, sizeof buf, "%s: \"wave hello\" stays in place (%.2f m) and lifts the right hand above the head (%+.2f m)", method, sv.forward, sv.wrist_over);
        check(std::abs(sv.forward) < 0.3 && sv.wrist_over > 0, buf);
        std::snprintf(buf, sizeof buf, "%s: \"raise both arms\" lifts both hands above the head (%+.2f m)", method, sa.arms_over);
        check(sa.arms_over > 0, buf);
        const GeneratedClip again = m->generate("walk forward", 1);
        check(again.root == walk.root && again.angles == walk.angles, std::string(method) + ": same text and seed, same motion");
        bool threw = false;
        try {
            m->generate("dance the tango");
        } catch (const std::invalid_argument& e) {
            threw = std::string(e.what()).find("Known words") != std::string::npos;
        }
        check(threw, std::string(method) + ": unknown words are reported with the known ones");
        if (std::string(method) == "diffusion") {
            const GeneratedClip fast = m->generate("walk forward", 1, 25);
            std::snprintf(buf, sizeof buf, "diffusion, 25 steps: still walks (%.2f m, %.0f ms)", summarize(*m, fast).forward, fast.ms);
            check(summarize(*m, fast).forward > 1.0, buf);
        }
        const std::string bvh = bvh_text(m->skeleton(), walk.root.data(), walk.angles.data(), walk.frames, m->fps());
        check(bvh.rfind("HIERARCHY", 0) == 0 && bvh.find("Frames: 90") != std::string::npos, std::string(method) + ": BVH text");
    }
    std::printf("%s\n", failures ? "SOME CHECKS FAILED" : "all passed");
    return failures ? 1 : 0;
}
