// Part specs: a small JSON language for simple mechanical parts that an LLM
// (even a ~1B model in the browser) can write reliably, and the C++ pipeline
// that turns a spec into a checked, printable mesh:
//
//   JSON spec -> validate -> SDF (CSG of primitives) -> grid -> marching cubes
//             -> printability checks -> feedback for the LLM
//
// Units are millimetres, Y is up, the part rests on the ground (y = 0).
#pragma once
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "vibecad/mesh.hpp"
#include "vibecad/printability.hpp"

namespace vibecad::csg {

enum class Shape { Box, Cylinder, Sphere, Torus, Capsule, Cone, Prism };

struct Part {
    Shape shape = Shape::Box;
    std::array<float, 3> size{};  // meaning depends on shape (see spec_guide())
    int sides = 6;                // prism only
    std::array<float, 3> at{};    // centre, mm
    std::array<float, 3> rotate{};// degrees, applied X then Y then Z
    bool cut = false;             // subtract instead of add
    float blend = 0;              // smooth blend radius, mm
    float round = 0;              // box edge rounding, mm
    int repeat = 1;               // copies around the vertical axis through the origin
    std::array<float, 3> color{0.62f, 0.64f, 0.68f};
    std::string label;            // "part 3 (cylinder)", for feedback

    // derived at parse time
    std::array<float, 9> world_to_local{};
    float sector = 0, sector_offset = 0;
};

struct Scene {
    std::string name;
    std::vector<Part> parts;
    std::vector<std::string> warnings;
};

// Parses and validates a spec. Throws SpecError listing every problem found.
struct SpecError : std::runtime_error {
    std::vector<std::string> problems;
    explicit SpecError(std::vector<std::string> p);
};
Scene parse_spec(const std::string& json);

// Signed distance (mm) of the whole part, and of one part alone (cuts ignored).
float distance(const Scene& scene, const Vec3& p);
float part_distance(const Part& part, const Vec3& p);

// Axis-aligned bounds of the solid (from the add parts), mm.
void bounds(const Scene& scene, Vec3& lo, Vec3& hi);

struct BuildOptions {
    int resolution = 112;     // grid cells along the longest side
    float min_wall = 1.0f;    // mm
    size_t target_triangles = 0;  // 0 = keep the full mesh
    PrintSettings print;
};

struct BuildResult {
    bool ok = false;                 // valid spec and a non-empty solid
    std::string name;
    std::vector<std::string> errors, warnings;
    Mesh mesh;                       // Y up, mm, as in the spec
    std::vector<Vec3> colors;        // per vertex, 0-1
    MeshStats stats;
    PrintReport print;
    ThicknessReport thickness;
    std::string verdict;             // printable | printable with supports | not printable
    std::vector<std::string> notes;  // human-readable printability findings
    std::string feedback;            // instructions for the LLM; empty when nothing to fix
    std::string expanded;            // the parts a family expanded to (empty for free-form specs)
    double ms_sample = 0, ms_mesh = 0, ms_check = 0;
};

BuildResult build(const std::string& json, const BuildOptions& options = {});

// The spec language, as text for an LLM prompt: part families first, then free-form parts.
std::string spec_guide();
std::string custom_guide();  // the free-form parts language only

// Binary STL in memory (for browsers and other callers without a file system).
std::vector<char> stl_bytes(const Mesh& mesh);

}  // namespace vibecad::csg
