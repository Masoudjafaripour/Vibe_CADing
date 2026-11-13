// mc_demo <sphere|torus|snowman> [cells per axis = 128] [out.stl]
// Samples an SDF, runs marching cubes, checks the mesh, writes binary STL.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "vibecad/mesh.hpp"
#include "vibecad/shapes.hpp"

using namespace vibecad;

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    const std::string shape = argc > 1 ? argv[1] : "snowman";
    const int n = argc > 2 ? std::atoi(argv[2]) : 128;
    const std::string out = argc > 3 ? argv[3] : shape + ".stl";
    if (n < 2) {
        std::fprintf(stderr, "cells per axis must be >= 2\n");
        return 2;
    }

    auto t0 = std::chrono::steady_clock::now();
    Grid grid;
    if (shape == "sphere")
        grid = sample([](float x, float y, float z) { return shapes::sphere(x, y, z, 0.6f); }, n);
    else if (shape == "torus")
        grid = sample([](float x, float y, float z) { return shapes::torus(x, y, z, 0.5f, 0.2f); }, n);
    else if (shape == "snowman")
        grid = sample(shapes::snowman, n);
    else {
        std::fprintf(stderr, "unknown shape '%s' (sphere | torus | snowman)\n", shape.c_str());
        return 2;
    }
    const double t_sample = ms_since(t0);

    t0 = std::chrono::steady_clock::now();
    const Mesh mesh = marching_cubes(grid);
    const double t_mc = ms_since(t0);

    t0 = std::chrono::steady_clock::now();
    const MeshStats s = analyze(mesh);
    const double t_stats = ms_since(t0);

    write_stl(mesh, out);

    std::printf("%s  %d^3 cells  ->  %zu vertices, %zu triangles\n", shape.c_str(), n,
                mesh.vertices.size(), mesh.triangles.size());
    std::printf("time: sample %.1f ms | marching cubes %.1f ms | checks %.1f ms\n", t_sample, t_mc, t_stats);
    std::printf("volume %.4f | area %.4f | watertight %s | oriented %s | shells %ld | genus %ld\n",
                s.volume, s.area, s.watertight ? "yes" : "no", s.oriented ? "yes" : "no", s.components, s.genus);
    std::printf("wrote %s\n", out.c_str());
    return s.watertight ? 0 : 1;
}
