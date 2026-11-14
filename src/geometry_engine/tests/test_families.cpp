// Every part family builds a closed, printable part with the topology it should have.
#include <cmath>
#include <cstdio>
#include <string>

#include "vibecad/csg.hpp"
#include "vibecad/families.hpp"

using namespace vibecad;
using namespace vibecad::csg;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? "pass" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
BuildResult make(const std::string& family, const std::string& params = "{}") {
    BuildOptions o;
    o.resolution = 144;
    return build("{\"name\": \"t\", \"family\": \"" + family + "\", \"params\": " + params + "}", o);
}
}  // namespace

int main() {
    const struct { const char* family; long genus; } expected[] = {
        {"flange", 7},     {"pulley", 1},         {"knurled_knob", 0}, {"star_knob", 0},
        {"hex_nut", 1},    {"hand_wheel", 6},     {"wall_hook", 2},    {"cable_organizer", 0},
        {"spur_gear", 1},  {"washer", 1},         {"standoff", 1},     {"l_bracket", 4},
    };
    check(families().size() == sizeof expected / sizeof expected[0], "every family is tested");
    for (const auto& e : expected) {
        const BuildResult r = make(e.family);
        const bool good = r.ok && r.stats.watertight && r.stats.components == 1 && r.verdict != "not printable";
        check(good && r.stats.genus == e.genus,
              std::string(e.family) + ": closed, printable (" + r.verdict + "), genus " + std::to_string(r.stats.genus) +
                  " (expected " + std::to_string(e.genus) + ")" + (r.feedback.empty() ? "" : " | " + r.feedback.substr(0, 120)));
    }

    // Parameters change the part: 8 bolt holes -> genus 9.
    check(make("flange", R"({"count": 8})").stats.genus == 9, "flange with 8 holes has genus 9");
    // Rim torus (1) + spokes - 1 independent loops + bore (1).
    check(make("hand_wheel", R"({"spokes": 3})").stats.genus == 4, "hand wheel with 3 spokes has genus 1 + 2 + 1 = 4");

    // ISO sizes: M8 nut is 13 mm across flats and 6.8 mm high.
    BuildResult r = make("hex_nut", R"({"thread": 8})");
    const double area = std::sqrt(3.0) / 2 * 13 * 13, hole = std::acos(-1.0) * 4 * 4;
    check(std::abs(r.stats.volume - (area - hole) * 6.8) / ((area - hole) * 6.8) < 0.03, "M8 nut: ISO 13 mm across flats, 6.8 mm high (volume within 3%)");
    check(r.name == "M8 hex nut", "nut name carries the thread size");

    // Out-of-range values are clamped and reported, not rejected.
    r = make("washer", R"({"outer": 900, "inner": 10})");
    check(r.ok && !r.warnings.empty() && r.warnings[0].find("outside") != std::string::npos, "out-of-range value clamped with a note");
    r = make("flange", R"({"spokes": 4})");
    check(r.ok && !r.warnings.empty() && r.warnings[0].find("does not apply") != std::string::npos, "irrelevant parameter ignored with a note");

    // Unknown family: a clear message; "custom" still takes free-form parts.
    r = build(R"({"name": "x", "family": "teapot"})");
    check(!r.ok && r.feedback.find("unknown family") != std::string::npos, "unknown family reported");
    r = build(R"({"name": "x", "family": "custom", "parts": [{"shape": "sphere", "size": [20], "at": [0, 10, 0]}]})");
    check(r.ok && r.expanded.empty(), "custom family uses the free-form parts");
    check(!make("spur_gear").expanded.empty(), "the expanded parts are returned for display");

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "all passed", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
