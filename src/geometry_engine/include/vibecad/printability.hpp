// 3D-printability analysis (FDM): wall thickness, overhangs, floating parts,
// bed contact, build volume. Mesh checks run on a Z-up mesh in millimetres;
// wall thickness runs on the SDF grid.
#pragma once
#include <string>
#include <vector>

#include "vibecad/mesh.hpp"

namespace vibecad {

struct ThicknessReport {
    float min_thickness = 0;   // thinnest wall found (grid units)
    float p05_thickness = 0;   // 5th percentile: robust "thin wall" value
    float median_thickness = 0;
    long ridge_voxels = 0;     // samples on the medial ridge (where thickness is measured)
    long thin_voxels = 0;      // ridge samples thinner than the threshold
    double thin_fraction = 0;  // thin_voxels / ridge_voxels
    Vec3 thinnest{};           // location of the thinnest wall (grid / SDF coordinates)
    Vec3 thin_centroid{};      // average location of the thin material (below)
    // Thin material by volume (morphological opening): solid that a ball of
    // diameter min_wall cannot reach. This is what decides printability.
    long solid_voxels = 0, thin_volume_voxels = 0;
    double thin_volume_fraction = 0;
};

// Local wall thickness of the solid (sdf < iso) via exact Euclidean distance
// transforms of the voxelised shape: thickness on the medial ridge, plus the
// volume of material thinner than min_wall (morphological opening with a ball of
// that diameter). Independent of whether the SDF is a true distance; accuracy is
// about one grid cell.
ThicknessReport wall_thickness(const Grid& grid, float min_wall, float iso = 0.f);

struct PrintSettings {
    double overhang_deg = 45;   // faces tilted further than this from vertical need support
    double plate_eps = 0.3;     // mm: "touching the build plate"
    double bed_x = 220, bed_y = 220, bed_z = 250;  // build volume, mm
    double min_contact = 25;    // mm^2 of flat bed contact before warning about adhesion...
    double min_contact_fraction = 0.03;  // ...or this share of the part's footprint (x * y)
};

struct Issue {
    std::string kind;      // not_watertight | floating | overhang | small_contact | too_big
    std::string severity;  // critical | warning
    std::string message;
    Vec3 where{};          // representative location, mm (print frame)
    double value = 0;
};

struct PrintReport {
    bool watertight = false;
    long shells = 0, floating_shells = 0;
    double volume_mm3 = 0, area_mm2 = 0;
    double overhang_area_mm2 = 0, overhang_fraction = 0;
    double contact_area_mm2 = 0;
    Vec3 size_mm{};
    std::vector<Issue> issues;
};

// Mesh-based checks; `mesh` is Z-up, in millimetres, resting on z = 0.
PrintReport check_print(const Mesh& mesh, const PrintSettings& settings = {});

// Merges duplicate positions of a triangle soup (e.g. an STL file) into an indexed mesh.
Mesh weld(const std::vector<Vec3>& soup_vertices);

// Reads a binary or ASCII STL file as an indexed (welded) mesh.
Mesh read_stl(const std::string& path);

}  // namespace vibecad
