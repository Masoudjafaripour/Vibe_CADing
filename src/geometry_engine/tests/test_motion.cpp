// Analytic checks for the motion core (Python parity is tested in
// src/animation/test_motion_fast.py against the original NumPy code).
#include <cmath>
#include <cstdio>
#include <string>

#include "vibecad/motion.hpp"

using namespace vibecad::motion;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? "pass" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
bool near(double a, double b, double tol = 1e-12) { return std::abs(a - b) < tol; }
}  // namespace

int main() {
    const double pi = std::acos(-1.0);
    // Two-bone chain: root -> elbow (up 1) -> wrist (up 1)
    Skeleton arm{{"root", "elbow", "wrist"}, {-1, 0, 1}, {Vec3{0, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 1, 0}}};
    double root[3] = {0.5, 0, 0};
    double angles[9] = {0, 0, pi / 2, 0, 0, 0, 0, 0, 0};  // root turned 90 deg about Z
    double pos[9];
    forward_kinematics(arm, root, angles, 1, pos);
    check(near(pos[3], -0.5) && near(pos[4], 0) && near(pos[6], -1.5), "FK: 90 deg about Z swings +Y to -X");

    angles[2] = 0;
    angles[3] = pi / 2;  // elbow bent 90 deg about X: wrist points to +Z
    forward_kinematics(arm, root, angles, 1, pos);
    check(near(pos[7], 1) && near(pos[8], 1), "FK: child rotation applies to the next bone only");

    // Constant-velocity motion has zero jerk.
    double line[4 * 3];
    for (int t = 0; t < 4; ++t) line[t * 3] = 0.1 * t, line[t * 3 + 1] = 0, line[t * 3 + 2] = 0;
    check(near(jerk(Clip{line, 4, 1}, 30), 0, 1e-9), "jerk of a straight line at constant speed is 0");

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "all passed", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
