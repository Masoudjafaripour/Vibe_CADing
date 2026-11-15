// Assemblies: mates place parts rigidly, "each" makes copies, collisions and bad mates are reported.
#include <cmath>
#include <cstdio>
#include <string>

#include "vibecad/assembly.hpp"

using namespace vibecad;
using namespace vibecad::csg;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? "pass" : "FAIL", what.c_str());
    if (!ok) ++failures;
}
std::string q(const std::string& s) {  // JSON string literal
    std::string o = "\"";
    for (char c : s) o += (c == '"' || c == '\\') ? std::string("\\") + c : std::string(1, c);
    return o + "\"";
}
const Placed* find(const AssemblyResult& r, const std::string& id) {
    for (const auto& p : r.placed)
        if (p.id == id) return &p;
    return nullptr;
}
bool near(float a, float b, float tol = 0.05f) { return std::abs(a - b) < tol; }
}  // namespace

int main() {
    const std::string flange = R"({"name": "flange", "family": "flange", "params": {"count": 4, "hole_diameter": 9, "thickness": 10}})";
    const std::string bolt = R"({"name": "bolt", "family": "bolt", "params": {"thread": 8, "length": 30}})";
    const std::string nut = R"({"name": "nut", "family": "hex_nut", "params": {"thread": 8}})";
    const std::string shaft = R"({"name": "shaft", "family": "shaft", "params": {"diameter": 8, "length": 60}})";
    const std::string wheel = R"({"name": "wheel", "family": "hand_wheel", "params": {"bore": 8.4}})";
    const std::string parts = "{\"p1\": " + q(flange) + ", \"p2\": " + q(bolt) + ", \"p3\": " + q(nut) + ", \"p4\": " + q(shaft) +
                              ", \"p5\": " + q(wheel) + "}";

    // Ports: families declare them, every part also has the bounding-box ones.
    const auto fp = part_ports(scene_of(flange));
    int holes = 0;
    for (const auto& p : fp) holes += p.name.rfind("hole_", 0) == 0 && p.name.size() > 5 && p.name.find("_top") != std::string::npos;
    check(holes == 4, "flange with 4 holes has hole_1_top..hole_4_top");
    check(describe_parts(parts).find("hole_1_top..hole_4_top") != std::string::npos, "describe_parts collapses numbered ports");

    // A bolt in every hole, a nut under each: 4 + 4 copies, no collisions with 9 mm holes and M8 bolts.
    AssemblyResult r = assemble(parts, R"({"name": "bolted flange", "mates": [
        {"part": "p2", "port": "under_head", "to": "p1", "to_port": "hole_1_top", "each": true},
        {"part": "p3", "port": "bore_top", "to": "p1", "to_port": "hole_1_bottom", "each": true}]})");
    check(r.ok && r.feedback.empty(), "bolted flange assembles: " + r.feedback.substr(0, 200));
    check(r.placed.size() == 9, "flange + 4 bolts + 4 nuts placed (" + std::to_string(r.placed.size()) + ")");
    check(r.overlaps.empty(), "no collisions (" + std::to_string(r.overlaps.size()) + ")");
    if (const Placed* b = find(r, "p2#1")) {
        // Flipped (head up): the bolt's local +y points down; under the head sits on the flange top (y = 10).
        check(near(b->rotation[4], -1), "bolt is turned head-up");
        const float under = b->rotation[4] * 0.65f * 8 + b->translation[1];  // world y of under_head (local y = k)
        check(near(under, 10, 0.01f), "bolt head rests on the flange top");
    } else {
        check(false, "bolt copy p2#1 exists");
    }
    if (const Placed* n = find(r, "p3#2")) check(near(n->translation[1], -6.8f, 0.05f), "nut hangs under the flange (top at y = 0)");

    // Wheel slid 15 mm onto a shaft standing on the flange.
    r = assemble(parts, R"({"mates": [
        {"part": "p4", "port": "bottom_end", "to": "p1", "to_port": "top"},
        {"part": "p5", "port": "bore_bottom", "to": "p4", "to_port": "top_end", "offset": -15}]})");
    const Placed* w = find(r, "p5");
    check(r.ok && w && near(w->translation[1], 10 + 60 - 15),
          "wheel sits 15 mm down the shaft (y = " + std::to_string(w ? w->translation[1] : -1) + ") " + r.feedback.substr(0, 200));
    check(r.overlaps.empty(), "8.4 mm bore on an 8 mm shaft: no collision");

    // "skip" leaves copies out; ids keep the hole number.
    r = assemble(parts, R"({"mates": [
        {"part": "p2", "port": "under_head", "to": "p1", "to_port": "hole_1_top", "each": true, "skip": [3]}]})");
    check(r.ok && r.placed.size() == 4 && find(r, "p2#4") && !find(r, "p2#3"), "skip [3]: bolts in holes 1, 2, 4 only");

    // Listing every hole after "each": true is redundant, not wrong: dropped with a note.
    r = assemble(parts, R"({"mates": [
        {"part": "p2", "port": "under_head", "to": "p1", "to_port": "hole_1_top", "each": true},
        {"part": "p2", "port": "under_head", "to": "p1", "to_port": "hole_2_top", "each": true}]})");
    check(r.ok && r.placed.size() == 5, "a repeated each-mate is ignored (" + std::to_string(r.placed.size()) + " placed)");

    // Errors the model can fix.
    r = assemble(parts, R"({"mates": [{"part": "p2", "port": "head_bottom", "to": "p1", "to_port": "hole_1_top"}]})");
    check(!r.ok && r.feedback.find("Its ports are") != std::string::npos && r.feedback.find("under_head") != std::string::npos,
          "unknown port lists the real ones");
    r = assemble(parts, R"({"mates": [{"part": "p9", "port": "top", "to": "p1", "to_port": "top"}]})");
    check(!r.ok && r.feedback.find("The parts are p1, p2") != std::string::npos, "unknown part lists the parts");
    r = assemble(parts, R"({"mates": [{"part": "p2", "port": "under_head", "to": "p1", "to_port": "hole_1_top"},
                                       {"part": "p3", "port": "bore_top", "to": "p4", "to_port": "top_end"}]})");
    check(!r.ok && r.feedback.find("not connected") != std::string::npos, "two separate groups are reported");
    // Shaft pushed into a solid plate (wrong offset sign): a collision the model must fix.
    const std::string plate = R"({"family": "washer", "params": {"outer": 40, "inner": 2, "thickness": 12}})";
    r = assemble("{\"a\": " + q(plate) + ", \"b\": " + q(shaft) + "}",
                 R"({"mates": [{"part": "b", "port": "bottom_end", "to": "a", "to_port": "top", "offset": -10}]})");
    check(!r.ok && r.feedback.find("pass through each other") != std::string::npos, "part pushed into another is reported");

    std::printf("%s\n", failures ? "SOME CHECKS FAILED" : "all passed");
    return failures ? 1 : 0;
}
