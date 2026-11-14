// Parametric part families. A language model picks a family and fills a few
// numbers ("M12" -> thread 12, "six bolt holes" -> count 6); the engine writes
// the CSG parts with correct proportions, sane limits and standard sizes.
// Small models are good at this and bad at 3D layout, so families give far
// better parts than free-form specs (see src/live_cad/README.md).
#pragma once
#include <string>
#include <vector>

#include "vibecad/json.hpp"

namespace vibecad::csg {

struct FamilyParam {
    std::string name;
    float value, lo, hi;  // default and allowed range
    bool integer;
    std::string doc;
};

struct FamilyInfo {
    std::string name, doc;
    std::vector<FamilyParam> params;
};

const std::vector<FamilyInfo>& families();

// {"family": "...", "params": {...}} -> parts spec JSON (the language in csg.hpp).
// Out-of-range values are clamped and reported in `notes`; problems throw SpecError.
std::string expand_family(const json::Value& root, std::vector<std::string>& notes);

}  // namespace vibecad::csg
