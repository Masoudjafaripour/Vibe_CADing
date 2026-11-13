// Python module `vibecad_geom`: NumPy in, NumPy out. Install with
//   pip install ./src/geometry_engine
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>

#include "vibecad/mesh.hpp"

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
        "write_stl",
        [](const FloatArray& vertices, const IndexArray& triangles, const std::string& path) {
            write_stl(to_mesh(vertices, triangles), path);
        },
        "vertices"_a, "triangles"_a, "path"_a, "Writes a binary STL file.");
}
