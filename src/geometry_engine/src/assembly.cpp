#include "vibecad/assembly.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>

#include "vibecad/families.hpp"
#include "vibecad/json.hpp"

namespace vibecad::csg {
namespace {

using Mat3 = std::array<float, 9>;  // row-major
constexpr Mat3 kIdentity{1, 0, 0, 0, 1, 0, 0, 0, 1};

std::string fmt(const char* f, double a = 0, double b = 0, double c = 0, double d = 0) {
    char buf[256];
    std::snprintf(buf, sizeof buf, f, a, b, c, d);
    return buf;
}

Vec3 mul(const Mat3& m, const Vec3& v) {
    return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}
Vec3 mul_t(const Mat3& m, const Vec3& v) {  // transpose (inverse rotation)
    return {m[0] * v[0] + m[3] * v[1] + m[6] * v[2], m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
            m[2] * v[0] + m[5] * v[1] + m[8] * v[2]};
}
Mat3 mul(const Mat3& a, const Mat3& b) {
    Mat3 c{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) c[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    return c;
}
float dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 cross(const Vec3& a, const Vec3& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }
Vec3 add(const Vec3& a, const Vec3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 scale(const Vec3& a, float s) { return {a[0] * s, a[1] * s, a[2] * s}; }
Vec3 normalize(const Vec3& a) {
    const float l = std::sqrt(dot(a, a));
    return l > 1e-9f ? scale(a, 1 / l) : Vec3{0, 1, 0};
}

// Rotation by `angle` (radians) about the unit axis k (Rodrigues).
Mat3 axis_angle(const Vec3& k, float angle) {
    const float c = std::cos(angle), s = std::sin(angle), t = 1 - c;
    return {t * k[0] * k[0] + c,        t * k[0] * k[1] - s * k[2], t * k[0] * k[2] + s * k[1],
            t * k[0] * k[1] + s * k[2], t * k[1] * k[1] + c,        t * k[1] * k[2] - s * k[0],
            t * k[0] * k[2] - s * k[1], t * k[1] * k[2] + s * k[0], t * k[2] * k[2] + c};
}

// The smallest rotation taking unit vector u to unit vector v. For opposite vectors, a half
// turn about an axis perpendicular to u (x if possible): a bolt standing up flips over x.
Mat3 align(const Vec3& u, const Vec3& v) {
    const float c = dot(u, v);
    if (c > 0.99999f) return kIdentity;
    if (c < -0.99999f) {
        const Vec3 helper = std::abs(u[0]) < 0.9f ? Vec3{1, 0, 0} : Vec3{0, 0, 1};
        return axis_angle(normalize(cross(cross(u, helper), u)), 3.14159265f);
    }
    return axis_angle(normalize(cross(u, v)), std::acos(std::clamp(c, -1.f, 1.f)));
}

// "hole_3_top" -> ("hole_", 3, "_top"); no number -> n = -1.
struct Numbered {
    std::string prefix, suffix;
    int n = -1;
};
Numbered split_number(const std::string& name) {
    Numbered r;
    size_t i = 0;
    while (i < name.size() && !std::isdigit(static_cast<unsigned char>(name[i]))) ++i;
    if (i == name.size()) return r;
    size_t j = i;
    while (j < name.size() && std::isdigit(static_cast<unsigned char>(name[j]))) ++j;
    r.prefix = name.substr(0, i), r.suffix = name.substr(j), r.n = std::stoi(name.substr(i, j - i));
    return r;
}

// Port names in short form: hole_1_top ... hole_6_top collapse to "hole_1_top..hole_6_top".
std::string port_list(const std::vector<Port>& ports) {
    std::vector<std::string> out;
    std::map<std::pair<std::string, std::string>, std::pair<int, int>> ranges;  // (prefix, suffix) -> (min, max)
    std::vector<std::pair<std::string, std::string>> order;
    for (const Port& p : ports) {
        const Numbered k = split_number(p.name);
        if (k.n < 0) {
            out.push_back(p.name);
            continue;
        }
        const auto key = std::make_pair(k.prefix, k.suffix);
        auto it = ranges.find(key);
        if (it == ranges.end()) {
            ranges[key] = {k.n, k.n};
            order.push_back(key);
            out.push_back("\x01" + std::to_string(order.size() - 1));  // placeholder, filled below
        } else {
            it->second.first = std::min(it->second.first, k.n), it->second.second = std::max(it->second.second, k.n);
        }
    }
    std::string s;
    for (const std::string& o : out) {
        std::string item = o;
        if (!o.empty() && o[0] == '\x01') {
            const auto& key = order[std::stoi(o.substr(1))];
            const auto [a, b] = ranges[key];
            item = key.first + std::to_string(a) + key.second;
            if (b > a) item += ".." + key.first + std::to_string(b) + key.second;
        }
        s += (s.empty() ? "" : ", ") + item;
    }
    return s;
}

struct Source {
    std::string id, spec, name;
    Scene scene;
    std::vector<Port> ports;
    Vec3 lo, hi;
    BuildResult mesh;  // built lazily
    bool built = false;
};

const Port* find_port(const Source& s, const std::string& name) {
    for (const Port& p : s.ports)
        if (p.name == name) return &p;
    return nullptr;
}

struct Instance {
    const Source* src;
    Placed placed;
};

float instance_distance(const Instance& in, const Vec3& q) {
    return distance(in.src->scene, mul_t(in.placed.rotation, sub(q, in.placed.translation)));
}

void world_bounds(const Instance& in, Vec3& lo, Vec3& hi) {
    lo = {1e9f, 1e9f, 1e9f}, hi = {-1e9f, -1e9f, -1e9f};
    for (int c = 0; c < 8; ++c) {
        const Vec3 corner{c & 1 ? in.src->hi[0] : in.src->lo[0], c & 2 ? in.src->hi[1] : in.src->lo[1],
                          c & 4 ? in.src->hi[2] : in.src->lo[2]};
        const Vec3 w = add(mul(in.placed.rotation, corner), in.placed.translation);
        for (int d = 0; d < 3; ++d) lo[d] = std::min(lo[d], w[d]), hi[d] = std::max(hi[d], w[d]);
    }
}

std::string describe_mate(size_t i, const std::string& part, const std::string& port, const std::string& to, const std::string& to_port) {
    return "mate " + std::to_string(i + 1) + " (" + part + "." + port + " to " + to + "." + to_port + ")";
}

}  // namespace

std::vector<Port> part_ports(const Scene& scene) {
    std::vector<Port> ports = scene.ports;
    Vec3 lo, hi;
    tight_bounds(scene, lo, hi);
    const Vec3 c{(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2};
    const std::pair<const char*, std::pair<Vec3, Vec3>> generic[] = {
        {"top", {{c[0], hi[1], c[2]}, {0, 1, 0}}},   {"bottom", {{c[0], lo[1], c[2]}, {0, -1, 0}}},
        {"right", {{hi[0], c[1], c[2]}, {1, 0, 0}}}, {"left", {{lo[0], c[1], c[2]}, {-1, 0, 0}}},
        {"front", {{c[0], c[1], hi[2]}, {0, 0, 1}}}, {"back", {{c[0], c[1], lo[2]}, {0, 0, -1}}}};
    for (const auto& [name, frame] : generic) {
        if (std::any_of(ports.begin(), ports.end(), [&](const Port& p) { return p.name == name; })) continue;
        ports.push_back(Port{name, frame.first, frame.second});
    }
    return ports;
}

Scene scene_of(const std::string& spec, std::vector<std::string>* notes) {
    std::string text = spec;
    json::Value root;
    try {
        root = json::parse(spec);
    } catch (const json::ParseError&) {
    }
    const json::Value* fam = root.get("family");
    if (fam && fam->is_string() && fam->string != "custom") {
        std::vector<std::string> n;
        text = expand_family(root, n);
        if (notes) notes->insert(notes->end(), n.begin(), n.end());
    }
    return parse_spec(text);
}

AssemblyResult assemble(const std::string& parts_json, const std::string& mates_json, const AssemblyOptions& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    AssemblyResult out;
    auto fail = [&](std::vector<std::string> errors) {
        out.errors = std::move(errors);
        std::string fb = "Your assembly has problems. Fix them and return the whole corrected JSON:";
        for (const auto& e : out.errors) fb += "\n- " + e;
        out.feedback = fb;
        out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return out;
    };

    // ---- the parts ----
    std::map<std::string, Source> sources;
    try {
        const json::Value parts = json::parse(parts_json);
        if (!parts.is_object()) return fail({"internal: parts must be an object of specs"});
        for (const auto& [id, v] : parts.object) {
            if (!v.is_string()) continue;
            Source s;
            s.id = id, s.spec = v.string;
            try {
                s.scene = scene_of(s.spec);
            } catch (const SpecError& e) {
                out.notes.push_back(id + " could not be built: " + (e.problems.empty() ? "" : e.problems[0]));
                continue;
            }
            s.name = s.scene.name.empty() ? id : s.scene.name;
            s.ports = part_ports(s.scene);
            tight_bounds(s.scene, s.lo, s.hi);
            sources.emplace(id, std::move(s));
        }
    } catch (const json::ParseError& e) {
        return fail({std::string("internal: parts JSON: ") + e.what()});
    }
    if (sources.empty()) return fail({"there are no parts to assemble yet: create some parts first"});
    std::string ids;
    for (const auto& [id, s] : sources) ids += (ids.empty() ? "" : ", ") + id;

    // ---- the mates ----
    json::Value root;
    try {
        root = json::parse(mates_json);
    } catch (const json::ParseError& e) {
        return fail({std::string(e.what()) + " (at character " + std::to_string(e.offset) + ")"});
    }
    if (const auto* n = root.get("name"); n && n->is_string()) out.name = n->string.substr(0, 80);
    const json::Value* mates = root.get("mates");
    if (!mates || !mates->is_array() || mates->array.empty())
        return fail({"the JSON needs a non-empty \"mates\" list, e.g. {\"part\": \"p2\", \"port\": \"bore_bottom\", \"to\": \"p1\", \"to_port\": \"top_end\"}"});

    struct Mate {
        std::string part, port, to, to_port;
        float offset = 0, angle = 0;
        bool each = false;
        size_t index = 0;  // position in the model's list, for messages
    };
    std::vector<Mate> list;
    std::vector<std::string> errors;
    auto str = [](const json::Value& o, const char* k) {
        const json::Value* v = o.get(k);
        return v && v->is_string() ? v->string : std::string();
    };
    for (size_t i = 0; i < mates->array.size() && i < 40; ++i) {
        const json::Value& m = mates->array[i];
        Mate mt{str(m, "part"), str(m, "port"), str(m, "to"), str(m, "to_port")};
        mt.index = i;
        if (const auto* v = m.get("offset"); v && v->is_number()) mt.offset = float(std::clamp(v->number, -500.0, 500.0));
        if (const auto* v = m.get("angle"); v && v->is_number()) mt.angle = float(v->number);
        if (const auto* v = m.get("each"); v && v->type == json::Value::Type::Bool) mt.each = v->boolean;
        const std::string where = describe_mate(i, mt.part, mt.port, mt.to, mt.to_port);
        bool ok = true;
        for (const auto& [id, port] : {std::pair{mt.part, mt.port}, std::pair{mt.to, mt.to_port}}) {
            auto it = sources.find(id);
            if (it == sources.end()) {
                errors.push_back(where + ": there is no part \"" + id + "\". The parts are " + ids + ".");
                ok = false;
            } else if (!find_port(it->second, port)) {
                errors.push_back(where + ": " + id + " (" + it->second.name + ") has no port \"" + port + "\". Its ports are: " +
                                 port_list(it->second.ports) + ".");
                ok = false;
            }
        }
        if (ok && mt.part == mt.to) {
            errors.push_back(where + ": a part cannot be attached to itself");
            ok = false;
        }
        if (ok && mt.each && split_number(mt.to_port).n < 0) {
            errors.push_back(where + ": \"each\" needs a numbered to_port (like hole_1_top), so that there is one copy per number");
            ok = false;
        }
        if (ok) {
            // A second mate for the same part into the same numbered group repeats an "each" mate.
            // Same port and offset: the intent is clear, so drop it with a note. Otherwise ask for one mate.
            const Numbered key = split_number(mt.to_port);
            for (size_t j = 0; j < list.size() && ok; ++j) {
                const Numbered other = split_number(list[j].to_port);
                if (list[j].part == mt.part && list[j].to == mt.to && (list[j].each || mt.each) && key.n >= 0 &&
                    other.prefix == key.prefix && other.suffix == key.suffix) {
                    ok = false;
                    if (list[j].port == mt.port && list[j].offset == mt.offset && list[j].angle == mt.angle) {
                        out.notes.push_back(where + " repeats an \"each\" mate and was ignored");
                        continue;
                    }
                    errors.push_back(where + " repeats an earlier mate: \"each\": true already puts one copy of " + mt.part +
                                     " at every " + key.prefix + "N" + key.suffix + ". Keep only one of these mates.");
                }
            }
        }
        if (ok) list.push_back(mt);
    }
    if (!errors.empty()) return fail(errors);

    // ---- placement: a tree grown from the fixed parts ----
    // A part that is never the "part" of a mate stays where it was designed (it is fixed).
    std::set<std::string> used, moved;
    for (const Mate& m : list) used.insert(m.part), used.insert(m.to), moved.insert(m.part);
    std::vector<std::string> roots;
    for (const std::string& id : used)
        if (!moved.count(id)) roots.push_back(id);
    if (roots.empty()) roots.push_back(list[0].to);  // every part moves: keep the first target still

    std::map<std::string, std::vector<Instance>> placed;  // id -> instances (several when copied with "each")
    for (const std::string& id : roots) {
        Instance in{&sources.at(id), {}};
        in.placed.id = in.placed.of = id, in.placed.name = in.src->name;
        placed[id].push_back(in);
    }
    std::vector<bool> applied(list.size(), false);
    // Where port `port` of `src` lands when attached to the world frame (pt, dir) of the target.
    auto attach = [](const Source& src, const Port& port, const Vec3& pt, const Vec3& dir, float offset, float angle_deg) {
        Placed pl;
        const Mat3 r = mul(axis_angle(dir, angle_deg * 3.14159265f / 180), align(port.dir, scale(dir, -1)));
        pl.rotation = r;
        pl.translation = sub(add(pt, scale(dir, offset)), mul(r, port.at));
        pl.name = src.name;
        return pl;
    };
    for (size_t pass = 0; pass < list.size() + 1; ++pass) {
        bool progress = false;
        for (size_t i = 0; i < list.size(); ++i) {
            const Mate& m = list[i];
            if (applied[i] || !placed.count(m.to)) continue;
            const std::string where = describe_mate(m.index, m.part, m.port, m.to, m.to_port);
            const auto& targets = placed.at(m.to);
            if (targets.size() > 1) {
                errors.push_back(where + ": " + m.to + " was copied with \"each\", so it is ambiguous which copy to use. "
                                 "Attach to a part that is not copied.");
                applied[i] = true;
                continue;
            }
            const Instance& target = targets[0];
            const Source& src = sources.at(m.part);
            const Port& port = *find_port(src, m.port);
            std::vector<const Port*> to_ports;
            if (m.each) {
                const Numbered key = split_number(m.to_port);
                std::vector<std::pair<int, const Port*>> numbered;
                for (const Port& p : target.src->ports) {
                    const Numbered k = split_number(p.name);
                    if (k.n >= 0 && k.prefix == key.prefix && k.suffix == key.suffix) numbered.push_back({k.n, &p});
                }
                std::sort(numbered.begin(), numbered.end(), [](auto& a, auto& b) { return a.first < b.first; });
                for (const auto& [n, p] : numbered) to_ports.push_back(p);
            } else {
                to_ports.push_back(find_port(*target.src, m.to_port));
            }
            std::vector<Instance> made;
            for (size_t c = 0; c < to_ports.size(); ++c) {
                const Vec3 pt = add(mul(target.placed.rotation, to_ports[c]->at), target.placed.translation);
                const Vec3 dir = normalize(mul(target.placed.rotation, to_ports[c]->dir));
                Instance in{&src, attach(src, port, pt, dir, m.offset, m.angle)};
                in.placed.of = m.part;
                in.placed.id = to_ports.size() > 1 ? m.part + "#" + std::to_string(c + 1) : m.part;
                made.push_back(in);
            }
            applied[i] = true;
            progress = true;
            if (!placed.count(m.part)) {
                placed[m.part] = made;
                continue;
            }
            // Placed already by an earlier mate: this one must agree (a second constraint).
            const auto& before = placed.at(m.part);
            if (before.size() != made.size()) {
                errors.push_back(where + " places " + m.part + " a second time, with a different number of copies. Use one mate per part.");
                continue;
            }
            for (size_t c = 0; c < made.size(); ++c) {
                const Vec3 a = add(mul(before[c].placed.rotation, port.at), before[c].placed.translation);
                const Vec3 b = add(mul(made[c].placed.rotation, port.at), made[c].placed.translation);
                const float gap = std::sqrt(dot(sub(a, b), sub(a, b)));
                const float turn = dot(mul(before[c].placed.rotation, port.dir), mul(made[c].placed.rotation, port.dir));
                if (gap > 0.5f || turn < 0.999f) {
                    errors.push_back(where + " disagrees with an earlier mate for " + m.part +
                                     fmt(" (%.1f mm apart). Use one mate per part.", gap));
                    break;
                }
            }
        }
        if (!progress) break;
    }
    for (size_t i = 0; i < list.size(); ++i)
        if (!applied[i])
            errors.push_back(describe_mate(list[i].index, list[i].part, list[i].port, list[i].to, list[i].to_port) + ": " + list[i].to +
                             " is never placed, so nothing can be attached to it. Attach " + list[i].to + " to a placed part first.");
    if (roots.size() > 1) {
        std::string r;
        for (const auto& id : roots) r += (r.empty() ? "" : ", ") + id;
        errors.push_back("these parts are not connected to each other: " + r +
                         ". Attach all but one of them to another part (each part except the base needs a mate).");
    }
    if (!errors.empty()) return fail(errors);

    // ---- meshes ----
    std::vector<Instance> all;
    for (auto& [id, ins] : placed)
        for (auto& in : ins) all.push_back(in);
    for (const Instance& in : all) {
        Source& s = sources.at(in.placed.of);
        if (!s.built) {
            BuildOptions bo;
            bo.resolution = opt.resolution, bo.target_triangles = opt.target_triangles;
            s.mesh = build(s.spec, bo);
            s.built = true;
        }
    }
    for (size_t k = 0; k < all.size(); ++k) {
        const Instance& in = all[k];
        const BuildResult& b = in.src->mesh;
        if (!b.ok) continue;
        const uint32_t base = uint32_t(out.mesh.vertices.size());
        for (size_t v = 0; v < b.mesh.vertices.size(); ++v) {
            out.mesh.vertices.push_back(add(mul(in.placed.rotation, b.mesh.vertices[v]), in.placed.translation));
            out.colors.push_back(b.colors[v]);
            out.part_of_vertex.push_back(uint32_t(k));
        }
        for (const auto& t : b.mesh.triangles) out.mesh.triangles.push_back({t[0] + base, t[1] + base, t[2] + base});
        out.placed.push_back(in.placed);
    }

    // ---- collisions: sample where the bounding boxes meet; inside both = interference ----
    for (size_t i = 0; i < all.size(); ++i)
        for (size_t j = i + 1; j < all.size(); ++j) {
            Vec3 alo, ahi, blo, bhi, lo, hi;
            world_bounds(all[i], alo, ahi);
            world_bounds(all[j], blo, bhi);
            bool empty = false;
            for (int d = 0; d < 3; ++d) {
                lo[d] = std::max(alo[d], blo[d]), hi[d] = std::min(ahi[d], bhi[d]);
                empty |= hi[d] <= lo[d];
            }
            if (empty) continue;
            const float extent = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
            const float step = std::max(extent / 32, 0.25f);
            int count = 0;
            Vec3 sum{};
            for (float x = lo[0] + step / 2; x < hi[0]; x += step)
                for (float y = lo[1] + step / 2; y < hi[1]; y += step)
                    for (float z = lo[2] + step / 2; z < hi[2]; z += step) {
                        const Vec3 q{x, y, z};
                        if (instance_distance(all[i], q) < -opt.tolerance && instance_distance(all[j], q) < -opt.tolerance)
                            ++count, sum = add(sum, q);
                    }
            const float volume = count * step * step * step;
            if (count < 2 || volume < 1) continue;
            out.overlaps.push_back({all[i].placed.id, all[j].placed.id, volume, scale(sum, 1.f / count)});
        }
    for (const Overlap& o : out.overlaps) {
        auto vol = [&](const std::string& id) {
            for (const Instance& in : all)
                if (in.placed.id == id) return std::max(in.src->mesh.stats.volume, 1.0);
            return 1.0;
        };
        const double fraction = o.volume_mm3 / std::min(vol(o.a), vol(o.b));
        if (fraction > 0.05 || o.volume_mm3 > 100) {  // more than a tight fit: the parts pass through each other
            auto name = [&](const std::string& id) {
                for (const Instance& in : all)
                    if (in.placed.id == id) return id + " (" + in.src->name + ")";
                return id;
            };
            errors.push_back(name(o.a) + " and " + name(o.b) +
                             fmt(" pass through each other (%.0f mm^3, near [%.0f, %.0f, %.0f]). ", o.volume_mm3, o.where[0], o.where[1], o.where[2]) +
                             "Check the ports: a part that a mate turns over has its top ports underneath. Check the offset "
                             "sign too: a negative offset moves a part into its target.");
        } else {
            out.notes.push_back(o.a + " and " + o.b + fmt(" interfere by %.0f mm^3 near [%.0f, %.0f, %.0f]", o.volume_mm3, o.where[0], o.where[1], o.where[2]) +
                                " (a shaft or bolt bigger than its hole?).");
        }
    }
    out.ok = errors.empty();
    if (!errors.empty()) {
        auto mesh = std::move(out.mesh);
        auto colors = std::move(out.colors);
        auto pv = std::move(out.part_of_vertex);
        auto pl = out.placed;
        fail(errors);
        out.mesh = std::move(mesh), out.colors = std::move(colors), out.part_of_vertex = std::move(pv), out.placed = pl;
        return out;  // still show the attempt
    }
    if (out.overlaps.empty()) out.notes.push_back("no collisions between parts");
    out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

std::string describe_parts(const std::string& parts_json) {
    std::string s;
    json::Value parts;
    try {
        parts = json::parse(parts_json);
    } catch (const json::ParseError&) {
        return s;
    }
    for (const auto& [id, v] : parts.object) {
        if (!v.is_string()) continue;
        try {
            const Scene sc = scene_of(v.string);
            Vec3 lo, hi;
            tight_bounds(sc, lo, hi);
            s += "- " + id + ": " + (sc.name.empty() ? "part" : sc.name) +
                 fmt(", %.0f x %.0f x %.0f mm. Ports: ", hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) + port_list(part_ports(sc)) + "\n";
        } catch (const SpecError&) {
        }
    }
    return s;
}

std::string assembly_guide() {
    return R"(Assemble the parts by writing mates. Answer with JSON:
{"name": string, "mates": [{"part": id, "port": port, "to": id, "to_port": port, "offset"?: mm, "angle"?: degrees, "each"?: true}]}

A mate moves "part" so that its "port" meets "to_port" of "to": the two points touch and the faces point at each other.
- Ports are named points on a part with an outward direction: bore_top / bore_bottom (the openings of a centre hole),
  hole_1_top (a bolt hole), top_end of a shaft, under_head and tip of a bolt, and top, bottom, left, right, front, back
  (faces of the bounding box) on every part.
- "offset" (mm) slides the part along the target port's direction: negative moves it into the target. To slide a wheel
  15 mm onto the top of a shaft: part = wheel, port = bore_bottom, to = shaft, to_port = top_end, offset = -15.
- A bolt goes through a hole from the top: port = under_head, to_port = hole_1_top. A nut goes on from below:
  port = bore_top, to_port = hole_1_bottom of the plate it clamps.
- "each": true repeats the mate for every port numbered like to_port (hole_1_top, hole_2_top, ...), with one copy of the
  part per port: "a bolt in every hole" is ONE mate with to_port = hole_1_top and "each": true. Do not list the other holes.
- Choose to_port by where the part goes: "on top of" a part uses its top port (bore_top, top_end, top), "under" or
  "below" uses its bottom port (bore_bottom, bottom_end, bottom).
- A shaft standing in a centre hole: part = shaft, port = bottom_end, to_port = bore_top, offset = minus the depth.
- "angle" turns the part about the target port's direction.
- The part that nothing is attached to stays in place (the base). Every other part needs exactly one mate, and the
  "to" of a mate must be the base or a part that is already attached.)";
}

}  // namespace vibecad::csg
