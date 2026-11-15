// WebAssembly entry points for the website's live text-to-CAD demo. Same
// engine as the Python module: part spec -> mesh -> printability -> feedback.
// Build: ./build.sh  (Emscripten)
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "vibecad/csg.hpp"
#include "vibecad/assembly.hpp"

using namespace emscripten;
using namespace vibecad;

namespace {

csg::BuildResult last;  // kept so stl() can export the last part without rebuilding

template <class T>
val copy_array(const char* type, const T* data, size_t n) {
    return val::global(type).new_(val(typed_memory_view(n, data)));
}

val strings(const std::vector<std::string>& v) {
    val out = val::array();
    for (const auto& s : v) out.call<void>("push", s);
    return out;
}

val build(const std::string& spec, int resolution, float min_wall, int target_triangles) {
    csg::BuildOptions o;
    o.resolution = resolution, o.min_wall = min_wall, o.target_triangles = size_t(std::max(target_triangles, 0));
    last = csg::build(spec, o);
    const csg::BuildResult& r = last;
    val out = val::object();
    out.set("ok", r.ok);
    out.set("name", r.name);
    out.set("verdict", r.verdict);
    out.set("feedback", r.feedback);
    out.set("expanded", r.expanded);
    out.set("errors", strings(r.errors));
    out.set("warnings", strings(r.warnings));
    out.set("notes", strings(r.notes));
    if (r.ok) {
        out.set("positions", copy_array("Float32Array", reinterpret_cast<const float*>(r.mesh.vertices.data()), r.mesh.vertices.size() * 3));
        out.set("indices", copy_array("Uint32Array", reinterpret_cast<const uint32_t*>(r.mesh.triangles.data()), r.mesh.triangles.size() * 3));
        out.set("colors", copy_array("Float32Array", reinterpret_cast<const float*>(r.colors.data()), r.colors.size() * 3));
    }
    out.set("triangles", double(r.mesh.triangles.size()));
    out.set("watertight", r.stats.watertight);
    out.set("shells", double(r.stats.components));
    out.set("genus", double(r.stats.genus));
    out.set("volume", r.stats.volume);
    val size = val::array();
    for (float s : r.print.size_mm) size.call<void>("push", s);
    out.set("size", size);
    out.set("overhang", r.print.overhang_fraction);
    out.set("thin", r.thickness.thin_volume_fraction);
    val ms = val::object();
    ms.set("sample", r.ms_sample), ms.set("mesh", r.ms_mesh), ms.set("check", r.ms_check);
    out.set("ms", ms);
    return out;
}

// Binary STL of the last built part (Z up, mm), as a Uint8Array.
val stl() {
    const std::vector<char> bytes = csg::stl_bytes(last.mesh);
    return copy_array("Uint8Array", reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
}

csg::AssemblyResult last_assembly;

// Parts ({id: spec}) + mates -> placed parts, collisions, merged mesh and feedback.
val assemble(const std::string& parts, const std::string& mates, int resolution) {
    csg::AssemblyOptions o;
    o.resolution = resolution;
    last_assembly = csg::assemble(parts, mates, o);
    const csg::AssemblyResult& r = last_assembly;
    val out = val::object();
    out.set("ok", r.ok);
    out.set("name", r.name);
    out.set("feedback", r.feedback);
    out.set("errors", strings(r.errors));
    out.set("notes", strings(r.notes));
    if (!r.mesh.triangles.empty()) {
        out.set("positions", copy_array("Float32Array", reinterpret_cast<const float*>(r.mesh.vertices.data()), r.mesh.vertices.size() * 3));
        out.set("indices", copy_array("Uint32Array", reinterpret_cast<const uint32_t*>(r.mesh.triangles.data()), r.mesh.triangles.size() * 3));
        out.set("colors", copy_array("Float32Array", reinterpret_cast<const float*>(r.colors.data()), r.colors.size() * 3));
    }
    val placed = val::array();
    for (const auto& p : r.placed) {
        val o2 = val::object();
        o2.set("id", p.id), o2.set("of", p.of), o2.set("name", p.name);
        placed.call<void>("push", o2);
    }
    out.set("placed", placed);
    out.set("overlaps", double(r.overlaps.size()));
    out.set("triangles", double(r.mesh.triangles.size()));
    out.set("ms", r.ms);
    return out;
}

// Port names of a part; empty if the spec does not build.
val part_ports(const std::string& spec) {
    std::vector<std::string> names;
    try {
        for (const auto& p : csg::part_ports(csg::scene_of(spec))) names.push_back(p.name);
    } catch (const csg::SpecError&) {
    }
    return strings(names);
}

val assembly_stl() {
    const std::vector<char> bytes = csg::stl_bytes(last_assembly.mesh);
    return copy_array("Uint8Array", reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
}

}  // namespace

EMSCRIPTEN_BINDINGS(vibecad) {
    function("assemble", &assemble);
    function("assemblyStl", &assembly_stl);
    function("describeParts", &csg::describe_parts);
    function("partPorts", &part_ports);
    function("assemblyGuide", &csg::assembly_guide);
    function("build", &build);
    function("stl", &stl);
    function("specGuide", &csg::spec_guide);
}
