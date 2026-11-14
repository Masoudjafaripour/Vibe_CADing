// Mesh processing: QEM simplification, Taubin smoothing, normals, glTF export.
#pragma once
#include <string>
#include <vector>

#include "vibecad/mesh.hpp"

namespace vibecad {

struct SimplifyReport {
    size_t triangles_before = 0, triangles_after = 0;
    size_t rejected_topology = 0;  // collapses refused to keep the mesh manifold (same genus / shells)
    size_t rejected_flip = 0;      // collapses refused because a face would flip or degenerate
};

// Quadric error metric edge collapse (Garland & Heckbert 1997) down to about
// `target_triangles`. Collapses that would break the link condition or flip a
// face are refused, so a closed mesh stays closed with the same genus and shells.
SimplifyReport simplify(Mesh& mesh, size_t target_triangles);

// Taubin lambda|mu smoothing: removes the small "terracing" of marching cubes
// without shrinking the shape the way plain Laplacian smoothing does.
void smooth_taubin(Mesh& mesh, int iterations, float lambda = 0.5f, float mu = -0.53f);

// Area-weighted vertex normals (unit length).
std::vector<Vec3> vertex_normals(const Mesh& mesh);

// Binary glTF 2.0 (.glb): positions, normals, optional per-vertex RGB colours
// (0-1), one triangle primitive. Readable by three.js, Blender, Unity, Godot.
void write_glb(const Mesh& mesh, const std::string& path, const std::vector<Vec3>& colors = {},
               const std::string& name = "mesh");

}  // namespace vibecad
