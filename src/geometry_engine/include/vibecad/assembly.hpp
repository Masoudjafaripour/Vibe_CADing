// Assemblies of parts from text. A language model writes *mates* between named
// ports ("put the wheel's bore_bottom on the shaft's top_end, 15 mm down"); the
// engine places the parts rigidly, checks collisions and explains problems:
//
//   parts (specs) + mates (JSON) -> ports -> placement (one rigid transform per part)
//                                -> overlap check on the distance fields -> mesh + feedback
//
// A mate puts port `port` of `part` against port `to_port` of `to`: the two points meet and
// the directions point at each other (faces touching). "offset" moves the part along the
// target port's direction (negative = into the target, e.g. sliding a wheel onto a shaft),
// "angle" turns it about that direction, and "each": true repeats the mate for every port
// numbered like to_port (hole_1_top -> hole_1_top ... hole_6_top), making copies of the part.
#pragma once
#include <array>
#include <string>
#include <vector>

#include "vibecad/csg.hpp"

namespace vibecad::csg {

// The ports of a part: the ones its spec declares (families write them), plus top, bottom,
// left, right, front and back at the centres of its bounding box faces.
std::vector<Port> part_ports(const Scene& scene);

// Builds the scene of a spec, expanding a family first (as build() does).
Scene scene_of(const std::string& spec, std::vector<std::string>* notes = nullptr);

struct Placed {
    std::string id;    // "p2", or "p2#3" for the third copy made by "each"
    std::string of;    // the part it is a copy of
    std::string name;
    std::array<float, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};  // local -> world, row-major
    Vec3 translation{};
};

struct Overlap {
    std::string a, b;
    float volume_mm3 = 0;
    Vec3 where{};
};

struct AssemblyOptions {
    int resolution = 80;          // mesh resolution of each part
    size_t target_triangles = 20000;  // per part, 0 = full mesh
    float tolerance = 0.15f;      // mm of interference allowed before two parts "overlap"
};

struct AssemblyResult {
    bool ok = false;
    std::string name;
    std::vector<std::string> errors;  // problems with the mates (sent back to the model)
    std::vector<std::string> notes;   // findings for the user
    std::string feedback;             // for the model; empty when nothing to fix
    std::vector<Placed> placed;
    std::vector<Overlap> overlaps;
    Mesh mesh;                        // all parts, world frame (Y up, mm)
    std::vector<Vec3> colors;         // per vertex
    std::vector<uint32_t> part_of_vertex;  // index into placed
    double ms = 0;
};

// parts_json: {"p1": "<spec>", "p2": "<spec>", ...} (each spec a JSON string, family or custom).
// mates_json: {"name": "...", "mates": [{"part", "port", "to", "to_port", "offset"?, "angle"?, "each"?}]}
AssemblyResult assemble(const std::string& parts_json, const std::string& mates_json, const AssemblyOptions& options = {});

// The parts as text for an LLM prompt: id, name, size and ports (numbered ports collapsed,
// "hole_1_top..hole_6_top").
std::string describe_parts(const std::string& parts_json);

// How to write mates, as text for an LLM prompt.
std::string assembly_guide();

}  // namespace vibecad::csg
