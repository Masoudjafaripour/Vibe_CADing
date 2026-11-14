// Part specs: parsing, validation messages, exact geometry, printability feedback.
#include <cmath>
#include <cstdio>
#include <string>

#include "vibecad/csg.hpp"

using namespace vibecad;
using namespace vibecad::csg;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? "pass" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
double rel(double a, double b) { return std::abs(a - b) / std::abs(b); }
bool has(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }
BuildOptions res(int n) {
    BuildOptions o;
    o.resolution = n;
    return o;
}
}  // namespace

int main() {
    const double pi = std::acos(-1.0);

    // A flange: 80 mm disc, 8 mm thick, 30 mm bore, six 8 mm bolt holes on a 60 mm circle.
    const std::string flange = R"({"name": "flange", "parts": [
        {"shape": "cylinder", "size": [80, 8], "at": [0, 4, 0], "color": "brass"},
        {"shape": "cylinder", "size": [30, 20], "at": [0, 4, 0], "op": "cut"},
        {"shape": "cylinder", "size": [8, 20], "at": [30, 4, 0], "op": "cut", "repeat": 6}]})";
    BuildResult r = build(flange, res(160));
    const double vol = pi * (40.0 * 40 - 15.0 * 15 - 6 * 4.0 * 4) * 8;
    check(r.ok && r.errors.empty(), "flange builds");
    check(r.stats.watertight && r.stats.genus == 7 && r.stats.components == 1, "flange genus 7 = bore + 6 bolt holes");
    check(rel(r.stats.volume, vol) < 0.02, "flange volume within 2% of the formula (" + std::to_string(int(r.stats.volume)) + " vs " + std::to_string(int(vol)) + ")");
    check(r.verdict == "printable" && r.feedback.empty(), "flange is printable, no feedback");
    check(r.colors.size() == r.mesh.vertices.size() && std::abs(r.colors[0][0] - 0.78f) < 1e-6, "vertices coloured brass");

    // Hexagonal prism: volume (sqrt(3)/2) F^2 h.
    r = build(R"({"parts": [{"shape": "prism", "size": [30, 12, 6], "at": [0, 6, 0]}]})", res(160));
    check(r.ok && rel(r.stats.volume, std::sqrt(3.0) / 2 * 30 * 30 * 12) < 0.02, "hex prism volume within 2%");

    // Rotated cylinder lies along x: its bounding box is long in x.
    r = build(R"({"parts": [{"shape": "cylinder", "size": [10, 60], "at": [0, 5, 0], "rotate": [0, 0, 90]}]})", res(128));
    float xmin = 1e9f, xmax = -1e9f, ymax = -1e9f;
    for (const Vec3& v : r.mesh.vertices) xmin = std::min(xmin, v[0]), xmax = std::max(xmax, v[0]), ymax = std::max(ymax, v[1]);
    check(r.ok && std::abs((xmax - xmin) - 60) < 1.5 && std::abs(ymax - 10) < 1, "rotate [0,0,90] lays a cylinder along x");

    // A floating part: feedback names it and gives its position.
    r = build(R"({"parts": [{"shape": "box", "size": [40, 10, 40], "at": [0, 5, 0]},
                           {"shape": "sphere", "size": [10], "at": [0, 40, 0]}]})", res(112));
    check(r.verdict == "not printable" && has(r.feedback, "part 2 (sphere) floats") && has(r.feedback, "y=35"),
          "floating sphere: feedback names part 2 and its height");

    // A thin fin: flagged with the part that causes it.
    r = build(R"({"parts": [{"shape": "box", "size": [40, 10, 40], "at": [0, 5, 0]},
                           {"shape": "box", "size": [0.6, 30, 30], "at": [0, 25, 0]}]})", res(160));
    check(r.verdict == "not printable" && has(r.feedback, "thinner than") && has(r.feedback, "part 2 (box)"),
          "0.6 mm fin: thin-wall feedback names part 2");

    // Validation messages are specific.
    r = build(R"({"parts": [{"shape": "cylinder", "size": [10], "at": [0, 5, 0]}, {"shape": "pyramid", "size": [5,5,5], "at": [0,0,0]}]})");
    check(!r.ok && has(r.feedback, "\"size\" must be a list of 2 numbers") && has(r.feedback, "got \"pyramid\""),
          "wrong size count and unknown shape are both reported");
    r = build(R"({"parts": [{"shape": "box", "size": [5,5,5], "at": [0,0,0], "op": "cut"}]})");
    check(!r.ok && has(r.feedback, "part 1 must be \"add\""), "a spec cannot start with a cut");
    r = build("here is the part: {\"parts\": [ {\"shape\": \"box\", ");
    check(!r.ok && has(r.feedback, "invalid JSON"), "truncated JSON is reported");
    r = build("```json\n{\"parts\": [{\"shape\": \"sphere\", \"size\": [20], \"at\": [0, 10, 0]}]}\n```");
    check(r.ok, "a ```json fence around the spec is accepted");

    // Semantic mistakes small models make, reported with a concrete fix.
    r = build(R"({"parts": [{"shape": "cylinder", "size": [80, 8], "at": [0, 4, 0]},
                           {"shape": "cylinder", "size": [8, 20], "at": [0, 4, 0], "op": "cut", "repeat": 6}]})");
    check(!r.ok && has(r.feedback, "all copies overlap") && has(r.feedback, "[R, 4, 0]"), "repeat on the axis: told to move it out");
    r = build(R"({"parts": [{"shape": "box", "size": [60, 10, 40], "at": [0, 5, 0]},
                           {"shape": "cylinder", "size": [6, 20], "at": [90, 5, 0], "op": "cut"}]})");
    check(!r.ok && has(r.feedback, "part 2 (cylinder) cuts nothing"), "a cut outside the part is reported");

    // Cuts that remove everything.
    r = build(R"({"parts": [{"shape": "box", "size": [10,10,10], "at": [0,5,0]}, {"shape": "box", "size": [30,30,30], "at": [0,5,0], "op": "cut"}]})");
    check(!r.ok && r.verdict == "empty" && !r.feedback.empty(), "everything cut away: reported as empty");

    // STL bytes: header + 50 bytes per triangle.
    r = build(flange, res(64));
    check(stl_bytes(r.mesh).size() == 84 + 50 * r.mesh.triangles.size(), "STL size matches the triangle count");
    check(!spec_guide().empty(), "spec guide available for prompts");

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "all passed", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
