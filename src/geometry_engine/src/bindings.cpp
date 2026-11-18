// Python module `vibecad_geom`: NumPy in, NumPy out. Install with
//   pip install ./src/geometry_engine
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "vibecad/mesh.hpp"
#include "vibecad/csg.hpp"
#include "vibecad/assembly.hpp"
#include "vibecad/families.hpp"
#include "vibecad/mesh_ops.hpp"
#include "vibecad/printability.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace vibecad;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

Mesh to_mesh(const FloatArray& vertices, const IndexArray& triangles) {
    if (vertices.ndim() != 2 || vertices.shape(1) != 3) throw py::value_error("vertices must have shape (N, 3)");
    if (triangles.ndim() != 2 || triangles.shape(1) != 3) throw py::value_error("triangles must have shape (M, 3)");
    Mesh mesh;
    mesh.vertices.resize(vertices.shape(0));
    mesh.triangles.resize(triangles.shape(0));
    std::memcpy(mesh.vertices.data(), vertices.data(), vertices.nbytes());
    std::memcpy(mesh.triangles.data(), triangles.data(), triangles.nbytes());
    for (const auto& t : mesh.triangles)
        for (uint32_t i : t)
            if (i >= mesh.vertices.size()) throw py::value_error("triangle index out of range");
    return mesh;
}

py::tuple from_mesh(const Mesh& mesh) {
    FloatArray v({py::ssize_t(mesh.vertices.size()), py::ssize_t(3)});
    IndexArray t({py::ssize_t(mesh.triangles.size()), py::ssize_t(3)});
    std::memcpy(v.mutable_data(), mesh.vertices.data(), v.nbytes());
    std::memcpy(t.mutable_data(), mesh.triangles.data(), t.nbytes());
    return py::make_tuple(v, t);
}

}  // namespace

PYBIND11_MODULE(vibecad_geom, m) {
    m.doc() = "SDF grid -> watertight triangle mesh (marching cubes), mesh checks, STL export.";

    m.def(
        "marching_cubes",
        [](const FloatArray& values, float lo, float hi, float iso) {
            if (values.ndim() != 3 || values.shape(0) != values.shape(1) || values.shape(1) != values.shape(2) ||
                values.shape(0) < 3)
                throw py::value_error("values must be a cube array of shape (m, m, m), m >= 3, indexed [z, y, x]");
            Grid grid{int(values.shape(0)) - 1, lo, hi,
                      std::vector<float>(values.data(), values.data() + values.size())};
            Mesh mesh;
            {
                py::gil_scoped_release release;  // pure C++ from here
                mesh = marching_cubes(grid, iso);
            }
            return from_mesh(mesh);
        },
        "values"_a, "lo"_a = -1.f, "hi"_a = 1.f, "iso"_a = 0.f,
        "SDF samples on an (m, m, m) grid over [lo, hi]^3 -> (vertices float32 (N, 3), triangles uint32 (M, 3)).");

    m.def(
        "analyze",
        [](const FloatArray& vertices, const IndexArray& triangles) {
            const MeshStats s = analyze(to_mesh(vertices, triangles));
            return py::dict("volume"_a = s.volume, "area"_a = s.area, "watertight"_a = s.watertight,
                            "oriented"_a = s.oriented, "components"_a = s.components, "genus"_a = s.genus,
                            "euler"_a = s.euler, "edges"_a = s.edges);
        },
        "vertices"_a, "triangles"_a, "Volume, area, watertight / oriented, shells and genus of a triangle mesh.");

    m.def(
        "wall_thickness",
        [](const FloatArray& values, float lo, float hi, float min_wall, float iso) {
            if (values.ndim() != 3 || values.shape(0) != values.shape(1) || values.shape(1) != values.shape(2))
                throw py::value_error("values must be a cube array of shape (m, m, m), indexed [z, y, x]");
            Grid grid{int(values.shape(0)) - 1, lo, hi, std::vector<float>(values.data(), values.data() + values.size())};
            ThicknessReport r;
            {
                py::gil_scoped_release release;
                r = wall_thickness(grid, min_wall, iso);
            }
            return py::dict("min"_a = r.min_thickness, "p05"_a = r.p05_thickness, "median"_a = r.median_thickness,
                            "ridge_samples"_a = r.ridge_voxels, "thin_samples"_a = r.thin_voxels,
                            "thin_fraction"_a = r.thin_fraction, "thinnest"_a = py::make_tuple(r.thinnest[0], r.thinnest[1], r.thinnest[2]),
                            "thin_centroid"_a = py::make_tuple(r.thin_centroid[0], r.thin_centroid[1], r.thin_centroid[2]),
                            "solid_samples"_a = r.solid_voxels, "thin_volume_samples"_a = r.thin_volume_voxels,
                            "thin_volume_fraction"_a = r.thin_volume_fraction);
        },
        "values"_a, "lo"_a = -1.f, "hi"_a = 1.f, "min_wall"_a = 0.02f, "iso"_a = 0.f,
        "Wall thickness (grid units) of the solid sdf < iso, from an exact distance transform.");

    m.def(
        "check_print",
        [](const FloatArray& vertices, const IndexArray& triangles, double overhang_deg, double plate_eps,
           std::array<double, 3> bed, double min_contact, double min_contact_fraction) {
            PrintSettings s;
            s.overhang_deg = overhang_deg, s.plate_eps = plate_eps, s.bed_x = bed[0], s.bed_y = bed[1], s.bed_z = bed[2];
            s.min_contact = min_contact, s.min_contact_fraction = min_contact_fraction;
            const PrintReport r = check_print(to_mesh(vertices, triangles), s);
            py::list issues;
            for (const Issue& i : r.issues)
                issues.append(py::dict("kind"_a = i.kind, "severity"_a = i.severity, "message"_a = i.message,
                                       "where"_a = py::make_tuple(i.where[0], i.where[1], i.where[2]), "value"_a = i.value));
            return py::dict("watertight"_a = r.watertight, "shells"_a = r.shells, "floating_shells"_a = r.floating_shells,
                            "volume_mm3"_a = r.volume_mm3, "area_mm2"_a = r.area_mm2,
                            "overhang_area_mm2"_a = r.overhang_area_mm2, "overhang_fraction"_a = r.overhang_fraction,
                            "contact_area_mm2"_a = r.contact_area_mm2,
                            "size_mm"_a = py::make_tuple(r.size_mm[0], r.size_mm[1], r.size_mm[2]), "issues"_a = issues);
        },
        "vertices"_a, "triangles"_a, "overhang_deg"_a = 45.0, "plate_eps"_a = 0.3,
        "bed"_a = std::array<double, 3>{220, 220, 250}, "min_contact"_a = 25.0, "min_contact_fraction"_a = 0.03,
        "3D-printability checks on a Z-up mesh in mm resting on z = 0: floating parts, overhangs, bed contact, size.");

    m.def(
        "read_stl",
        [](const std::string& path) { return from_mesh(read_stl(path)); }, "path"_a,
        "Reads a binary or ASCII STL as a welded mesh -> (vertices (N, 3), triangles (M, 3)).");

    m.def(
        "simplify",
        [](const FloatArray& vertices, const IndexArray& triangles, size_t target_triangles) {
            Mesh mesh = to_mesh(vertices, triangles);
            SimplifyReport r;
            {
                py::gil_scoped_release release;
                r = simplify(mesh, target_triangles);
            }
            return py::make_tuple(from_mesh(mesh)[0], from_mesh(mesh)[1],
                                  py::dict("triangles_before"_a = r.triangles_before, "triangles_after"_a = r.triangles_after,
                                           "rejected_topology"_a = r.rejected_topology, "rejected_flip"_a = r.rejected_flip));
        },
        "vertices"_a, "triangles"_a, "target_triangles"_a,
        "QEM edge-collapse simplification that keeps a closed mesh closed (same genus and shells).");

    m.def(
        "smooth_taubin",
        [](const FloatArray& vertices, const IndexArray& triangles, int iterations, float lambda_, float mu) {
            Mesh mesh = to_mesh(vertices, triangles);
            smooth_taubin(mesh, iterations, lambda_, mu);
            return from_mesh(mesh)[0];
        },
        "vertices"_a, "triangles"_a, "iterations"_a = 10, "lambda_"_a = 0.5f, "mu"_a = -0.53f,
        "Taubin smoothing (denoising without shrinking); returns new vertices.");

    m.def(
        "vertex_normals",
        [](const FloatArray& vertices, const IndexArray& triangles) {
            const std::vector<Vec3> n = vertex_normals(to_mesh(vertices, triangles));
            FloatArray out({py::ssize_t(n.size()), py::ssize_t(3)});
            std::memcpy(out.mutable_data(), n.data(), out.nbytes());
            return out;
        },
        "vertices"_a, "triangles"_a);

    m.def(
        "write_glb",
        [](const FloatArray& vertices, const IndexArray& triangles, const std::string& path, py::object colors,
           const std::string& name) {
            const Mesh mesh = to_mesh(vertices, triangles);
            std::vector<Vec3> c;
            if (!colors.is_none()) {
                const FloatArray ca = colors.cast<FloatArray>();
                if (ca.ndim() != 2 || ca.shape(1) != 3 || size_t(ca.shape(0)) != mesh.vertices.size())
                    throw py::value_error("colors must have shape (N, 3), one per vertex");
                c.resize(mesh.vertices.size());
                std::memcpy(c.data(), ca.data(), ca.nbytes());
            }
            write_glb(mesh, path, c, name);
        },
        "vertices"_a, "triangles"_a, "path"_a, "colors"_a = py::none(), "name"_a = "mesh",
        "Binary glTF 2.0 (.glb) with normals and optional per-vertex RGB colours (0-1).");

    m.def(
        "csg_build",
        [](const std::string& spec, int resolution, float min_wall, size_t target_triangles) {
            csg::BuildOptions o;
            o.resolution = resolution, o.min_wall = min_wall, o.target_triangles = target_triangles;
            csg::BuildResult r;
            {
                py::gil_scoped_release release;
                r = csg::build(spec, o);
            }
            py::object v = py::none(), t = py::none(), c = py::none();
            if (r.ok) {
                py::tuple vt = from_mesh(r.mesh);
                v = vt[0], t = vt[1];
                FloatArray ca({py::ssize_t(r.colors.size()), py::ssize_t(3)});
                std::memcpy(ca.mutable_data(), r.colors.data(), ca.nbytes());
                c = ca;
            }
            return py::dict("ok"_a = r.ok, "name"_a = r.name, "verdict"_a = r.verdict, "errors"_a = r.errors,
                            "expanded"_a = r.expanded,
                            "warnings"_a = r.warnings, "notes"_a = r.notes, "feedback"_a = r.feedback,
                            "vertices"_a = v, "triangles"_a = t, "colors"_a = c,
                            "watertight"_a = r.stats.watertight, "shells"_a = r.stats.components, "genus"_a = r.stats.genus,
                            "volume_mm3"_a = r.stats.volume, "floating_shells"_a = r.print.floating_shells,
                            "overhang_fraction"_a = r.print.overhang_fraction, "thin_fraction"_a = r.thickness.thin_volume_fraction,
                            "size_mm"_a = py::make_tuple(r.print.size_mm[0], r.print.size_mm[1], r.print.size_mm[2]),
                            "ms"_a = py::make_tuple(r.ms_sample, r.ms_mesh, r.ms_check));
        },
        "spec"_a, "resolution"_a = 112, "min_wall"_a = 1.0f, "target_triangles"_a = 0,
        "JSON part spec -> validated, meshed, printability-checked part (with feedback for an LLM).");

    m.def("spec_guide", &csg::spec_guide, "The part-spec language, as text for an LLM prompt.");

    m.def(
        "csg_assemble",
        [](const std::string& parts, const std::string& mates, int resolution, size_t target_triangles) {
            csg::AssemblyOptions o;
            o.resolution = resolution, o.target_triangles = target_triangles;
            csg::AssemblyResult r;
            {
                py::gil_scoped_release release;
                r = csg::assemble(parts, mates, o);
            }
            py::object v = py::none(), t = py::none(), c = py::none();
            if (!r.mesh.triangles.empty()) {
                py::tuple vt = from_mesh(r.mesh);
                v = vt[0], t = vt[1];
                FloatArray ca({py::ssize_t(r.colors.size()), py::ssize_t(3)});
                std::memcpy(ca.mutable_data(), r.colors.data(), ca.nbytes());
                c = ca;
            }
            py::list placed, overlaps;
            for (const auto& p : r.placed)
                placed.append(py::dict("id"_a = p.id, "of"_a = p.of, "name"_a = p.name, "rotation"_a = p.rotation,
                                       "translation"_a = p.translation));
            for (const auto& o2 : r.overlaps)
                overlaps.append(py::dict("a"_a = o2.a, "b"_a = o2.b, "volume_mm3"_a = o2.volume_mm3, "where"_a = o2.where));
            return py::dict("ok"_a = r.ok, "name"_a = r.name, "errors"_a = r.errors, "notes"_a = r.notes,
                            "feedback"_a = r.feedback, "placed"_a = placed, "overlaps"_a = overlaps,
                            "vertices"_a = v, "triangles"_a = t, "colors"_a = c, "ms"_a = r.ms);
        },
        "parts"_a, "mates"_a, "resolution"_a = 80, "target_triangles"_a = 20000,
        "Parts ({id: spec JSON}) + mates JSON -> placed parts, collision check, merged mesh, feedback for an LLM.");
    m.def(
        "rank_families",
        [](const std::string& text) {
            std::vector<std::pair<std::string, float>> out;
            for (const auto& r : csg::rank_families(text)) out.push_back({r.name, r.score});
            return out;
        },
        "text"_a, "Every part family with its keyword score for a request, best first (v2 prompts).");
    m.def("select_families", &csg::select_families, "text"_a, "k"_a = 4, "always"_a = "",
          "Up to k family names for a request (all of them when nothing matches).");
    m.def("families_guide", &csg::families_guide, "names"_a, "Prompt text for just these families.");
    m.def("describe_parts", &csg::describe_parts, "Parts ({id: spec JSON}) as text for an LLM prompt: name, size, ports.");
    m.def("assembly_guide", &csg::assembly_guide, "How to write mates, as text for an LLM prompt.");
    m.def(
        "part_ports",
        [](const std::string& spec) {
            std::vector<std::string> names;
            for (const auto& p : csg::part_ports(csg::scene_of(spec))) names.push_back(p.name);
            return names;
        },
        "spec"_a, "Port names of a part (declared by its family plus the bounding-box ones).");

    m.def(
        "write_stl",
        [](const FloatArray& vertices, const IndexArray& triangles, const std::string& path) {
            write_stl(to_mesh(vertices, triangles), path);
        },
        "vertices"_a, "triangles"_a, "path"_a, "Writes a binary STL file.");
}
