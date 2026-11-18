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

// ---- v2 ("fast") prompts: send the model only the families a request is about ----
struct FamilyScore {
    std::string name;
    float score = 0;
};
// Ranks every family by word overlap between the request and the family's name, synonyms
// ("Also: coat hook, ..."), description and parameter names. Words are lower-cased and
// singularized ("holes" -> "hole"); "M8"-style sizes count for the threaded families.
std::vector<FamilyScore> rank_families(const std::string& request);
// Up to k family names for a request, best first; `always` (e.g. the family of a part being
// edited) is always included. Returns every family when no word matches.
std::vector<std::string> select_families(const std::string& request, int k = 4, const std::string& always = "");
// The prompt text for those families: the answer format and one line per family with its params
// (the family half of spec_guide(), without "name" and without the free-form parts language).
std::string families_guide(const std::vector<std::string>& names);

// {"family": "...", "params": {...}} -> parts spec JSON (the language in csg.hpp).
// Out-of-range values are clamped and reported in `notes`; problems throw SpecError.
std::string expand_family(const json::Value& root, std::vector<std::string>& notes);

}  // namespace vibecad::csg
