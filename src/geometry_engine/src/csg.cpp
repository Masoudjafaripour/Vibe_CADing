#include "vibecad/csg.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>

#include "vibecad/families.hpp"
#include "vibecad/json.hpp"
#include "vibecad/mesh_ops.hpp"

namespace vibecad::csg {
namespace {

constexpr float kPi = 3.14159265358979f;
constexpr float kMaxSize = 200, kMaxReach = 120;  // mm

const std::map<std::string, Shape> kShapes = {
    {"box", Shape::Box},         {"cylinder", Shape::Cylinder}, {"sphere", Shape::Sphere}, {"torus", Shape::Torus},
    {"capsule", Shape::Capsule}, {"cone", Shape::Cone},         {"prism", Shape::Prism}};
// How many size values each shape takes.
int size_count(Shape s) {
    switch (s) {
        case Shape::Sphere: return 1;
        case Shape::Cylinder: case Shape::Torus: case Shape::Capsule: return 2;
        default: return 3;  // box, cone, prism (across flats, height, sides)
    }
}

const std::map<std::string, std::array<float, 3>> kColors = {
    {"steel", {0.62f, 0.64f, 0.68f}}, {"aluminium", {0.78f, 0.8f, 0.82f}}, {"brass", {0.78f, 0.6f, 0.25f}},
    {"copper", {0.72f, 0.42f, 0.28f}}, {"black", {0.13f, 0.13f, 0.15f}},  {"white", {0.92f, 0.92f, 0.9f}},
    {"red", {0.8f, 0.2f, 0.15f}},      {"orange", {0.95f, 0.5f, 0.15f}},  {"yellow", {0.95f, 0.8f, 0.25f}},
    {"green", {0.25f, 0.6f, 0.35f}},   {"blue", {0.2f, 0.42f, 0.78f}},    {"wood", {0.6f, 0.42f, 0.25f}}};

std::string fmt(const char* f, double a = 0, double b = 0, double c = 0, double d = 0) {
    char buf[256];
    std::snprintf(buf, sizeof buf, f, a, b, c, d);
    return buf;
}

// ---- primitive distances (local frame, Y up) ----

float length2(float x, float y) { return std::sqrt(x * x + y * y); }
float length3(float x, float y, float z) { return std::sqrt(x * x + y * y + z * z); }

float sd_box(float x, float y, float z, float hx, float hy, float hz, float r) {
    const float qx = std::abs(x) - (hx - r), qy = std::abs(y) - (hy - r), qz = std::abs(z) - (hz - r);
    return length3(std::max(qx, 0.f), std::max(qy, 0.f), std::max(qz, 0.f)) + std::min(std::max(qx, std::max(qy, qz)), 0.f) - r;
}

// Extrudes a 2D distance along y (half height h).
float extrude(float d2, float y, float h) {
    const float wy = std::abs(y) - h;
    return std::min(std::max(d2, wy), 0.f) + length2(std::max(d2, 0.f), std::max(wy, 0.f));
}

// Capped cone along y from radius r1 at -h to radius r2 at +h (Inigo Quilez).
float sd_cone(float x, float y, float z, float h, float r1, float r2) {
    const float qx = length2(x, z), qy = y;
    const float k1x = r2, k1y = h, k2x = r2 - r1, k2y = 2 * h;
    const float cax = qx - std::min(qx, qy < 0 ? r1 : r2), cay = std::abs(qy) - h;
    const float t = std::clamp(((k1x - qx) * k2x + (k1y - qy) * k2y) / (k2x * k2x + k2y * k2y), 0.f, 1.f);
    const float cbx = qx - k1x + k2x * t, cby = qy - k1y + k2y * t;
    const float s = (cbx < 0 && cay < 0) ? -1.f : 1.f;
    return s * std::sqrt(std::min(cax * cax + cay * cay, cbx * cbx + cby * cby));
}

// Regular n-gon (apothem a) in the xz plane, extruded along y.
float sd_prism(float x, float y, float z, float apothem, float h, int n) {
    const float sector = 2 * kPi / n;
    float a = std::atan2(z, x);
    a = std::fmod(a + sector * 0.5f + 64 * kPi, sector) - sector * 0.5f;  // fold into one sector
    const float r = length2(x, z);
    const float qx = r * std::cos(a), qy = r * std::sin(a);
    const float half_edge = apothem * std::tan(kPi / n);
    const float dx = qx - apothem, dy = std::max(std::abs(qy) - half_edge, 0.f);
    const float d2 = dx > 0 ? length2(dx, dy) : std::max(dx, -apothem);  // inside: distance to the nearest edge
    return extrude(d2, y, h);
}

float sd_local(const Part& p, float x, float y, float z) {
    const auto& s = p.size;
    switch (p.shape) {
        case Shape::Box: return sd_box(x, y, z, s[0] / 2, s[1] / 2, s[2] / 2, p.round);
        case Shape::Cylinder: return extrude(length2(x, z) - s[0] / 2, y, s[1] / 2);
        case Shape::Sphere: return length3(x, y, z) - s[0] / 2;
        case Shape::Torus: return length2(length2(x, z) - s[0] / 2, y) - s[1] / 2;
        case Shape::Capsule: {
            const float half = std::max(s[1] / 2 - s[0] / 2, 0.f);
            return length3(x, y - std::clamp(y, -half, half), z) - s[0] / 2;
        }
        case Shape::Cone: return sd_cone(x, y, z, s[2] / 2, s[0] / 2, s[1] / 2);
        case Shape::Prism: return sd_prism(x, y, z, s[0] / 2, s[1] / 2, p.sides);
    }
    return 1e9f;
}

float smin(float a, float b, float k) {
    const float h = std::clamp(0.5f + 0.5f * (b - a) / k, 0.f, 1.f);
    return b + (a - b) * h - k * h * (1 - h);
}
float smax(float a, float b, float k) { return -smin(-a, -b, k); }

// Radius of a sphere around the primitive's centre that contains it.
float bounding_radius(const Part& p) {
    const auto& s = p.size;
    switch (p.shape) {
        case Shape::Box: return length3(s[0], s[1], s[2]) / 2;
        case Shape::Cylinder: case Shape::Prism: {
            const float r = p.shape == Shape::Prism ? s[0] / 2 / std::cos(kPi / p.sides) : s[0] / 2;
            return length2(r, s[1] / 2);
        }
        case Shape::Sphere: return s[0] / 2;
        case Shape::Torus: return (s[0] + s[1]) / 2;
        case Shape::Capsule: return std::max(s[0], s[1]) / 2;
        case Shape::Cone: return length2(std::max(s[0], s[1]) / 2, s[2] / 2);
    }
    return 0;
}

// world -> local: R^T (p - at), with R = Rz Ry Rx (X applied first)
std::array<float, 9> inverse_rotation(const std::array<float, 3>& deg) {
    const float ax = deg[0] * kPi / 180, ay = deg[1] * kPi / 180, az = deg[2] * kPi / 180;
    const float cx = std::cos(ax), sx = std::sin(ax), cy = std::cos(ay), sy = std::sin(ay), cz = std::cos(az), sz = std::sin(az);
    // R = Rz * Ry * Rx
    const float r[9] = {cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
                        sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
                        -sy,     cy * sx,                cy * cx};
    return {r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]};  // transpose
}

// ---- validation helpers ----

struct Reader {
    std::vector<std::string>& problems;
    std::string where;

    bool numbers(const json::Value* v, const char* key, int n, std::array<float, 3>& out, bool required) {
        if (!v) {
            if (required) problems.push_back(where + ": missing \"" + key + "\"");
            return false;
        }
        if (!v->is_array() || int(v->array.size()) != n) {
            problems.push_back(where + ": \"" + key + "\" must be a list of " + std::to_string(n) + " numbers");
            return false;
        }
        for (int i = 0; i < n; ++i) {
            if (!v->array[i].is_number() || !std::isfinite(v->array[i].number)) {
                problems.push_back(where + ": \"" + key + "\" must contain only numbers");
                return false;
            }
            out[i] = float(v->array[i].number);
        }
        return true;
    }
};

}  // namespace

SpecError::SpecError(std::vector<std::string> p) : std::runtime_error(p.empty() ? "invalid spec" : p[0]), problems(std::move(p)) {}

Scene parse_spec(const std::string& text) {
    json::Value root;
    try {
        root = json::parse(text);
    } catch (const json::ParseError& e) {
        throw SpecError({std::string(e.what()) + " (at character " + std::to_string(e.offset) + ")"});
    }
    std::vector<std::string> problems;
    Scene scene;
    if (!root.is_object()) throw SpecError({"the spec must be a JSON object with a \"parts\" list"});
    if (const auto* n = root.get("name"); n && n->is_string()) scene.name = n->string.substr(0, 80);
    const auto* parts = root.get("parts");
    if (!parts || !parts->is_array() || parts->array.empty())
        throw SpecError({"the spec needs a non-empty \"parts\" list"});
    if (parts->array.size() > 24) problems.push_back("use at most 24 parts (got " + std::to_string(parts->array.size()) + ")");

    std::array<float, 3> default_color = kColors.at("steel");
    for (size_t i = 0; i < parts->array.size() && i < 24; ++i) {
        const json::Value& pv = parts->array[i];
        Part p;
        Reader r{problems, "part " + std::to_string(i + 1)};
        if (!pv.is_object()) {
            problems.push_back(r.where + ": must be an object");
            continue;
        }
        const auto* sh = pv.get("shape");
        if (!sh || !sh->is_string() || !kShapes.count(sh->string)) {
            problems.push_back(r.where + ": \"shape\" must be one of box, cylinder, sphere, torus, capsule, cone, prism" +
                               (sh && sh->is_string() ? " (got \"" + sh->string + "\")" : ""));
            continue;
        }
        p.shape = kShapes.at(sh->string);
        p.label = r.where + " (" + sh->string + ")";
        r.where = p.label;

        const int n = size_count(p.shape);
        if (r.numbers(pv.get("size"), "size", n, p.size, true)) {
            if (p.shape == Shape::Prism) {
                p.sides = int(std::lround(p.size[2]));
                if (p.sides < 3 || p.sides > 12) problems.push_back(r.where + ": a prism needs 3 to 12 sides (size[2])");
            }
            for (int k = 0; k < (p.shape == Shape::Prism ? 2 : n); ++k)
                if (!(p.size[k] > 0.2f) || p.size[k] > kMaxSize)
                    problems.push_back(r.where + fmt(": size values must be between 0.2 and %.0f mm", kMaxSize));
            if (p.shape == Shape::Torus && p.size[1] >= p.size[0])
                problems.push_back(r.where + ": a torus needs tube diameter (size[1]) smaller than ring diameter (size[0])");
        }
        r.numbers(pv.get("at"), "at", 3, p.at, true);
        for (float c : p.at)
            if (std::abs(c) > kMaxReach) problems.push_back(r.where + fmt(": \"at\" must stay within +-%.0f mm", kMaxReach));
        r.numbers(pv.get("rotate"), "rotate", 3, p.rotate, false);

        if (const auto* op = pv.get("op")) {
            if (op->is_string() && (op->string == "add" || op->string == "cut")) p.cut = op->string == "cut";
            else problems.push_back(r.where + ": \"op\" must be \"add\" or \"cut\"");
        }
        if (const auto* b = pv.get("blend"); b && b->is_number()) p.blend = float(std::clamp(b->number, 0.0, 20.0));
        if (const auto* rd = pv.get("round"); rd && rd->is_number() && p.shape == Shape::Box)
            p.round = std::clamp(float(rd->number), 0.f, std::min({p.size[0], p.size[1], p.size[2]}) / 2 * 0.95f);
        if (const auto* rp = pv.get("repeat"); rp && rp->is_number()) {
            p.repeat = int(std::lround(rp->number));
            if (p.repeat < 1 || p.repeat > 96) problems.push_back(r.where + ": \"repeat\" must be 1 to 96");
            else if (p.repeat > 1 && length2(p.at[0], p.at[2]) < 0.5f)
                problems.push_back(r.where + ": \"repeat\" makes copies around the vertical axis, but this part sits on "
                                   "the axis, so all copies overlap. Move it out to the circle radius, e.g. \"at\": [R, " +
                                   fmt("%.0f", p.at[1]) + ", 0] with R > 0.");
        }
        if (const auto* c = pv.get("color"); c && c->is_string()) {
            if (kColors.count(c->string)) p.color = kColors.at(c->string);
            else scene.warnings.push_back(r.where + ": unknown color \"" + c->string + "\" ignored");
        } else {
            p.color = default_color;
        }
        if (!p.cut && i == 0) default_color = p.color;  // uncoloured parts follow the first one

        p.world_to_local = inverse_rotation(p.rotate);
        if (p.repeat > 1) {
            p.sector = 2 * kPi / p.repeat;
            p.sector_offset = std::atan2(p.at[2], p.at[0]);
        }
        scene.parts.push_back(p);
    }
    if (!scene.parts.empty() && scene.parts[0].cut) problems.push_back("part 1 must be \"add\" (there is nothing to cut from yet)");
    if (std::none_of(scene.parts.begin(), scene.parts.end(), [](const Part& p) { return !p.cut; }))
        problems.push_back("at least one part must be \"add\"");
    if (problems.empty()) {
        Scene so_far;
        for (const Part& p : scene.parts) {
            if (p.cut && !so_far.parts.empty()) {
                // Does the cut overlap the material? Both are distance fields, so if their overlap is
                // non-empty some grid sample lies within half a cell diagonal of it: then
                // max(cut, material) < 0.87 * spacing there. Thin overlaps cannot slip between samples.
                const float rb = bounding_radius(p);
                constexpr int kSteps = 16;
                const float spacing = 2 * rb / kSteps;
                bool touches = false;
                for (int i = 0; i <= kSteps && !touches; ++i)
                    for (int j = 0; j <= kSteps && !touches; ++j)
                        for (int k = 0; k <= kSteps && !touches; ++k) {
                            const Vec3 q{p.at[0] - rb + spacing * i, p.at[1] - rb + spacing * j, p.at[2] - rb + spacing * k};
                            touches = std::max(part_distance(p, q), distance(so_far, q)) < 0.87f * spacing;
                        }
                if (!touches)
                    problems.push_back(p.label + fmt(" cuts nothing: it does not overlap the material at [%.0f, %.0f, %.0f]. "
                                                     "Move it into the part (a hole must pass through the solid).",
                                                     p.at[0], p.at[1], p.at[2]));
            }
            so_far.parts.push_back(p);
        }
    }
    if (!problems.empty()) throw SpecError(problems);
    return scene;
}

float part_distance(const Part& p, const Vec3& q) {
    float x = q[0], y = q[1], z = q[2];
    if (p.repeat > 1) {  // fold into the sector holding the part's own position
        const float a = std::atan2(z, x) - p.sector_offset;
        const float folded = std::fmod(a + p.sector * 0.5f + 64 * kPi, p.sector) - p.sector * 0.5f + p.sector_offset;
        const float r = length2(x, z);
        x = r * std::cos(folded);
        z = r * std::sin(folded);
    }
    x -= p.at[0], y -= p.at[1], z -= p.at[2];
    const auto& m = p.world_to_local;
    return sd_local(p, m[0] * x + m[1] * y + m[2] * z, m[3] * x + m[4] * y + m[5] * z, m[6] * x + m[7] * y + m[8] * z);
}

float distance(const Scene& scene, const Vec3& q) {
    float d = 1e9f;
    for (const Part& p : scene.parts) {
        const float dp = part_distance(p, q);
        if (p.cut) d = p.blend > 0 ? smax(d, -dp, p.blend) : std::max(d, -dp);
        else d = p.blend > 0 ? smin(d, dp, p.blend) : std::min(d, dp);
    }
    return d;
}

void bounds(const Scene& scene, Vec3& lo, Vec3& hi) {
    lo = {1e9f, 1e9f, 1e9f};
    hi = {-1e9f, -1e9f, -1e9f};
    for (const Part& p : scene.parts) {
        if (p.cut) continue;
        const float rb = bounding_radius(p) + p.blend;
        const float rho = p.repeat > 1 ? length2(p.at[0], p.at[2]) : 0;
        const float cx = p.repeat > 1 ? 0 : p.at[0], cz = p.repeat > 1 ? 0 : p.at[2];
        const float ext = rho + rb;
        lo = {std::min(lo[0], cx - ext), std::min(lo[1], p.at[1] - rb), std::min(lo[2], cz - ext)};
        hi = {std::max(hi[0], cx + ext), std::max(hi[1], p.at[1] + rb), std::max(hi[2], cz + ext)};
    }
}

namespace {

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// The add part closest to a point: used to colour vertices and to name parts in feedback.
const Part* nearest_part(const Scene& scene, const Vec3& q) {
    const Part* best = nullptr;
    float bd = 1e9f;
    for (const Part& p : scene.parts) {
        if (p.cut) continue;
        const float d = std::abs(part_distance(p, q));
        if (d < bd) bd = d, best = &p;
    }
    return best;
}

}  // namespace

BuildResult build(const std::string& input, const BuildOptions& opt) {
    BuildResult out;
    Scene scene;
    std::string text = input;
    try {
        // A family spec ({"family": "flange", "params": {...}}) expands to parts first.
        json::Value root;
        try {
            root = json::parse(input);
        } catch (const json::ParseError&) {
        }
        const json::Value* fam = root.get("family");
        if (fam && fam->is_string() && fam->string != "custom") {
            std::vector<std::string> notes;
            out.expanded = text = expand_family(root, notes);
            out.warnings = notes;
        }
        scene = parse_spec(text);
    } catch (const SpecError& e) {
        out.errors = e.problems;
        std::string fb = "Your JSON spec has problems. Fix them and return the whole corrected spec:";
        for (const auto& p : e.problems) fb += "\n- " + p;
        out.feedback = fb;
        out.verdict = "invalid spec";
        return out;
    }
    out.name = scene.name;
    out.warnings.insert(out.warnings.end(), scene.warnings.begin(), scene.warnings.end());

    // Sample on a cube around the solid (centred, a few cells of margin).
    Vec3 lo, hi;
    bounds(scene, lo, hi);
    const Vec3 c{(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2};
    const float extent = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
    const int n = std::clamp(opt.resolution, 16, 320);
    const float half = extent / 2 * (1 + 4.0f / n) + 0.5f;

    auto t0 = std::chrono::steady_clock::now();
    Grid grid = sample([&](float x, float y, float z) { return distance(scene, Vec3{x + c[0], y + c[1], z + c[2]}); }, n, -half, half);
    // Close the surface where it touches the sampling box.
    const int m = n + 1;
    for (int k = 0; k < m; ++k)
        for (int j = 0; j < m; ++j)
            for (int i = 0; i < m; ++i)
                if (i == 0 || j == 0 || k == 0 || i == n || j == n || k == n) {
                    float& v = grid.values[(size_t(k) * m + j) * m + i];
                    v = std::max(v, 1e-3f);
                }
    out.ms_sample = ms_since(t0);

    t0 = std::chrono::steady_clock::now();
    out.mesh = marching_cubes(grid);
    if (out.mesh.triangles.empty()) {
        out.verdict = "empty";
        out.feedback = "The spec produces no solid: the cuts remove everything, or the parts are too thin to see. "
                       "Make the \"add\" parts larger than the \"cut\" parts and return the whole corrected spec.";
        out.errors.push_back("empty solid");
        return out;
    }
    if (opt.target_triangles && out.mesh.triangles.size() > opt.target_triangles) simplify(out.mesh, opt.target_triangles);
    for (Vec3& v : out.mesh.vertices)
        for (int d = 0; d < 3; ++d) v[d] += c[d];
    out.ms_mesh = ms_since(t0);
    out.ok = true;

    t0 = std::chrono::steady_clock::now();
    out.stats = analyze(out.mesh);
    out.colors.resize(out.mesh.vertices.size());
    for (size_t i = 0; i < out.mesh.vertices.size(); ++i) {
        const Part* p = nearest_part(scene, out.mesh.vertices[i]);
        out.colors[i] = p ? Vec3{p->color[0], p->color[1], p->color[2]} : Vec3{0.62f, 0.64f, 0.68f};
    }

    // Printability on the print frame: Z up, resting on z = 0.
    float ymin = 1e9f;
    for (const Vec3& v : out.mesh.vertices) ymin = std::min(ymin, v[1]);
    Mesh printed = out.mesh;
    for (Vec3& v : printed.vertices) v = {v[0], -v[2], v[1] - ymin};
    out.print = check_print(printed, opt.print);
    out.thickness = wall_thickness(grid, opt.min_wall);
    auto to_spec = [&](const Vec3& w) { return Vec3{w[0], w[2] + ymin, -w[1]}; };
    out.ms_check = ms_since(t0);

    // Verdict + notes + LLM feedback (critical problems only).
    bool critical = false, overhang = false;
    std::vector<std::string> fixes;
    for (const Issue& i : out.print.issues) {
        out.notes.push_back(i.message);
        if (i.kind == "overhang") overhang = true;
        if (i.severity != "critical") continue;
        critical = true;
        if (i.kind == "floating") {
            const Vec3 at = to_spec(i.where);
            const Part* p = nearest_part(scene, at);
            fixes.push_back((p ? p->label : std::string("a part")) +
                            fmt(" floats in the air (lowest point x=%.0f, y=%.0f, z=%.0f mm): it is not connected to the rest. "
                                "Move it so it overlaps the body, or rest it on the ground (y = 0).", at[0], at[1], at[2]));
        } else {
            fixes.push_back(i.message);
        }
    }
    const double thin = out.thickness.thin_volume_fraction;
    if (thin > 0.001) {
        const Vec3 tc{out.thickness.thin_centroid[0] + c[0], out.thickness.thin_centroid[1] + c[1], out.thickness.thin_centroid[2] + c[2]};
        const Part* p = nearest_part(scene, tc);
        const std::string msg = fmt("%.1f%% of the material is thinner than %.1f mm", 100 * thin, opt.min_wall) +
                                (p ? " (mostly " + p->label + ")" : "") + "; thin features may break or not print.";
        out.notes.push_back(msg);
        if (thin > 0.02) {
            critical = true;
            fixes.push_back(msg + fmt(" Make walls and pins at least %.1f mm thick.", opt.min_wall * 1.5));
        }
    }
    out.verdict = critical ? "not printable" : overhang ? "printable with supports" : "printable";
    if (critical) {
        std::string fb = "The part builds but is not 3D-printable. Fix this and return the whole corrected spec:";
        for (const auto& f : fixes) fb += "\n- " + f;
        out.feedback = fb;
    }
    return out;
}

std::string spec_guide() {
    std::string g =
        "Answer with JSON. If the part matches one of these families, use it and fill the numbers from the request:\n"
        "{\"name\": string, \"family\": <family>, \"params\": {<param>: number, ...}}\n"
        "Leave out params you have no value for (defaults are sensible). Units are millimetres.\n\n";
    for (const auto& f : families()) {
        g += "- " + f.name + ": " + f.doc + "\n  params: ";
        for (size_t i = 0; i < f.params.size(); ++i) {
            const auto& p = f.params[i];
            char buf[200];
            std::snprintf(buf, sizeof buf, "%s%s (%s, default %g)", i ? "; " : "", p.name.c_str(), p.doc.c_str(), p.value);
            g += buf;
        }
        g += "\n";
    }
    g += "\nOnly if no family fits, use \"family\": \"custom\" with a \"parts\" list:\n\n";
    return g + custom_guide();
}

std::string custom_guide() {
    return R"(Describe the part as JSON: {"name": string, "parts": [part, ...]}.
Units are millimetres. Y is up; the part rests on the ground at y = 0. Keep it within about 120 mm.
Each part: {"shape", "size", "at", "rotate"?, "op"?, "blend"?, "round"?, "repeat"?, "color"?}
- "shape" and "size" (all in mm):
  box: [width x, height y, depth z]      cylinder: [diameter, height]     sphere: [diameter]
  torus: [ring diameter, tube diameter]   capsule: [diameter, length]     cone: [bottom diameter, top diameter, height]
  prism: [across-flats width, height, number of sides]   (6 sides = hexagon)
- "at": [x, y, z] centre of the shape. A shape of height h sitting on the ground has y = h / 2.
- Cylinders, capsules, cones and prisms stand upright (along y). "rotate": [rx, ry, rz] in degrees turns them;
  [0, 0, 90] lays one along x, [90, 0, 0] along z. A torus lies flat; [90, 0, 0] stands it up.
- "op": "add" (default) joins the shape, "cut" removes it. Parts apply in order; the first must be "add".
  To make a hole, "cut" a cylinder longer than the material it goes through.
- "blend": radius in mm for a smooth fillet with the parts before it (try 1 to 4).
- "round": rounds the edges of a box (mm).
- "repeat": n copies evenly around the vertical axis through x = 0, z = 0 (bolt circles, teeth, spokes, ribs).
- "color": steel, aluminium, brass, copper, black, white, red, orange, yellow, green, blue or wood.
Every add part must touch or overlap the others: nothing may float. Keep walls at least 2 mm thick.

Recipes (use real proportions; most hand-sized parts are 20 to 100 mm):
- Hole through a plate: a "cut" cylinder at the hole position, taller than the plate (rotate it if the plate stands up).
- Bolt circle / evenly spaced holes, teeth, ridges, spokes, lobes: ONE part placed at the circle radius,
  e.g. "at": [30, y, 0], with "repeat": N. Never put a repeated part at x = 0, z = 0.
- Centre bore: a "cut" cylinder at x = 0, z = 0, longer than the part.
- Groove around a wheel (V-belt pulley): a "cut" torus with ring diameter = wheel diameter.
- Hex nut or bolt head: a prism with 6 sides, plus a "cut" cylinder through the middle.
- Knurled or ridged knob: a cylinder, plus small "cut" cylinders (or "add" boxes) at its rim with "repeat".
- Spoked wheel: rim = torus, hub = cylinder, ONE spoke = box from the hub to the rim with "repeat".
- Hook or arm sticking out: a capsule or box rotated to point sideways, touching the base.)";
}

std::vector<char> stl_bytes(const Mesh& mesh) {
    std::vector<char> buf(84 + 50 * mesh.triangles.size(), 0);
    std::strncpy(buf.data(), "Vibe-CADing", 80);
    const uint32_t count = uint32_t(mesh.triangles.size());
    std::memcpy(buf.data() + 80, &count, 4);
    char* p = buf.data() + 84;
    for (const auto& t : mesh.triangles) {
        // Z-up for slicers: (x, y, z) Y-up -> (x, -z, y)
        float v[3][3];
        for (int k = 0; k < 3; ++k) {
            const Vec3& a = mesh.vertices[t[k]];
            v[k][0] = a[0], v[k][1] = -a[2], v[k][2] = a[1];
        }
        const float u[3] = {v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]};
        const float w[3] = {v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]};
        float nrm[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
        const float len = std::sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
        if (len > 0) for (float& x : nrm) x /= len;
        std::memcpy(p, nrm, 12);
        std::memcpy(p + 12, v, 36);
        p += 50;
    }
    return buf;
}

}  // namespace vibecad::csg
