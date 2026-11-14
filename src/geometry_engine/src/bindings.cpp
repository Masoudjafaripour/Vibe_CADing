// Python module `vibecad_geom`: NumPy in, NumPy out. Install with
//   pip install ./src/geometry_engine
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "vibecad/mesh.hpp"
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
        "write_stl",
        [](const FloatArray& vertices, const IndexArray& triangles, const std::string& path) {
            write_stl(to_mesh(vertices, triangles), path);
        },
        "vertices"_a, "triangles"_a, "path"_a, "Writes a binary STL file.");
}
