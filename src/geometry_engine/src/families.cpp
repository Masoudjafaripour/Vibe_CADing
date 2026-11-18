#include "vibecad/families.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <sstream>

#include "vibecad/csg.hpp"

namespace vibecad::csg {
namespace {

constexpr float kPi = 3.14159265358979f;
using Params = std::map<std::string, float>;

// Writes parts in the spec language.
class Parts {
public:
    explicit Parts(std::string color) : color_(std::move(color)) {}

    Parts& add(const char* shape, std::initializer_list<float> size, float x, float y, float z) {
        return part(shape, size, x, y, z, "add");
    }
    Parts& cut(const char* shape, std::initializer_list<float> size, float x, float y, float z) {
        return part(shape, size, x, y, z, "cut");
    }
    // Modifiers for the most recent part.
    Parts& rotate(float rx, float ry, float rz) { return extra(fmt("\"rotate\": [%g, %g, %g]", rx, ry, rz)); }
    Parts& repeat(int n) { return n > 1 ? extra(fmt("\"repeat\": %g", n)) : *this; }
    Parts& blend(float b) { return b > 0 ? extra(fmt("\"blend\": %g", b)) : *this; }
    Parts& round(float r) { return r > 0 ? extra(fmt("\"round\": %g", r)) : *this; }
    Parts& color(const char* c) { return extra(std::string("\"color\": \"") + c + "\""); }
    // A named connection point for assemblies: position and outward direction.
    Parts& port(const std::string& name, float x, float y, float z, float dx, float dy, float dz) {
        ports_.push_back("{\"name\": \"" + name + "\", \"at\": [" + num(x) + ", " + num(y) + ", " + num(z) +
                         "], \"dir\": [" + num(dx) + ", " + num(dy) + ", " + num(dz) + "]}");
        return *this;
    }
    // The usual pair for a part with a vertical bore: its top and bottom openings.
    Parts& bore_ports(float top) { return port("bore_top", 0, top, 0, 0, 1, 0).port("bore_bottom", 0, 0, 0, 0, -1, 0); }

    std::string json(const std::string& name) const {
        std::string s = "{\"name\": \"" + name + "\", \"parts\": [";
        for (size_t i = 0; i < parts_.size(); ++i) s += (i ? ",\n  " : "\n  ") + parts_[i] + "}";
        s += "\n]";
        if (!ports_.empty()) {
            s += ", \"ports\": [";
            for (size_t i = 0; i < ports_.size(); ++i) s += (i ? ",\n  " : "\n  ") + ports_[i];
            s += "\n]";
        }
        return s + "}";
    }

private:
    std::string color_;
    std::vector<std::string> parts_, ports_;

    static std::string fmt(const char* f, double a, double b = 0, double c = 0) {
        char buf[160];
        std::snprintf(buf, sizeof buf, f, a, b, c);
        return buf;
    }
    static std::string num(float v) { return fmt("%.4g", v); }

    Parts& part(const char* shape, std::initializer_list<float> size, float x, float y, float z, const char* op) {
        std::string s = std::string("{\"shape\": \"") + shape + "\", \"size\": [";
        bool first = true;
        for (float v : size) s += (first ? "" : ", ") + num(v), first = false;
        s += "], \"at\": [" + num(x) + ", " + num(y) + ", " + num(z) + "], \"op\": \"" + op + "\"";
        if (parts_.empty()) s += ", \"color\": \"" + color_ + "\"";
        parts_.push_back(s);
        return *this;
    }
    Parts& extra(const std::string& kv) {
        parts_.back() += ", " + kv;
        return *this;
    }
};

// ISO 4032 hex nuts: thread -> (across flats, height), mm.
float nut_across_flats(float m) {
    static const std::map<float, float> af = {{2, 4},    {2.5f, 5},   {3, 5.5f}, {4, 7},   {5, 8},   {6, 10},
                                              {8, 13},   {10, 16},    {12, 18},  {14, 21}, {16, 24}, {20, 30},
                                              {24, 36},  {30, 46},    {36, 55},  {42, 65}, {48, 75}};
    auto it = af.lower_bound(m - 0.01f);
    return (it != af.end() && std::abs(it->first - m) < 0.01f) ? it->second : 1.5f * m + 1;
}
float nut_height(float m) { return m <= 3 ? 0.8f * m : 0.85f * m; }

// ISO 261 coarse thread pitch, mm.
float iso_pitch(float m) {
    static const std::map<float, float> pitch = {{3, 0.5f}, {4, 0.7f}, {5, 0.8f}, {6, 1},     {8, 1.25f}, {10, 1.5f},
                                                 {12, 1.75f}, {14, 2},  {16, 2},    {20, 2.5f}, {24, 3}};
    auto it = pitch.lower_bound(m - 0.01f);
    return it != pitch.end() ? it->second : 3;
}

using Generator = std::function<std::string(Params&, std::vector<std::string>&)>;

struct Family {
    FamilyInfo info;
    Generator make;
};

const std::vector<Family>& table() {
    static const std::vector<Family> t = {
        {{"flange", "Round pipe flange: a disc with a centre bore and a circle of bolt holes, optional raised hub. Also: mounting disc, bolt-circle plate.",
          {{"diameter", 80, 20, 200, false, "outer diameter"},
           {"thickness", 10, 3, 40, false, "disc thickness"},
           {"bore", 30, 0, 150, false, "centre hole diameter (0 = none)"},
           {"count", 6, 0, 24, true, "number of bolt holes"},
           {"hole_diameter", 8, 2, 30, false, "bolt hole diameter"},
           {"hole_circle", 0, 0, 190, false, "bolt circle diameter (0 = automatic)"},
           {"hub_diameter", 0, 0, 150, false, "raised hub diameter (0 = none)"},
           {"hub_height", 10, 2, 60, false, "hub height above the disc"}}},
         [](Params& p, std::vector<std::string>& notes) {
             float D = p["diameter"], T = p["thickness"], B = std::min(p["bore"], D - 12), h = p["hole_diameter"];
             float hub = p["hub_diameter"] > B + 6 ? std::min(p["hub_diameter"], D - 8) : 0;
             const float inner = std::max(B, hub) / 2 + h / 2 + 2, outer = D / 2 - h / 2 - 2;
             float C = p["hole_circle"] / 2;
             if (C <= 0) C = (inner + outer) / 2;
             if (C < inner || C > outer) {
                 notes.push_back("hole_circle adjusted so the bolt holes sit inside the flange");
                 C = std::clamp(C, inner, std::max(inner, outer));
             }
             Parts s("steel");
             s.add("cylinder", {D, T}, 0, T / 2, 0);
             const float top = hub ? T + p["hub_height"] : T;
             if (hub) s.add("cylinder", {hub, top}, 0, top / 2, 0).blend(2);
             if (B > 0) s.cut("cylinder", {B, top + 10}, 0, top / 2, 0);
             if (p["count"] >= 1) s.cut("cylinder", {h, T + 10}, C, T / 2, 0).repeat(int(p["count"]));
             s.bore_ports(top);
             for (int i = 0; i < int(p["count"]); ++i) {  // same angles as "repeat": 2 pi i / n from +x towards +z
                 const float a = 2 * kPi * i / p["count"], x = C * std::cos(a), z = C * std::sin(a);
                 s.port("hole_" + std::to_string(i + 1) + "_top", x, T, z, 0, 1, 0);
                 s.port("hole_" + std::to_string(i + 1) + "_bottom", x, 0, z, 0, -1, 0);
             }
             return s.json("flange");
         }},

        {{"pulley", "V-belt pulley: a wheel with a groove around the rim, a centre bore and a hub. Also: belt wheel, sheave, rope or idler pulley.",
          {{"diameter", 60, 20, 200, false, "outer diameter"},
           {"width", 18, 8, 60, false, "wheel width"},
           {"groove_depth", 6, 2, 25, false, "depth of the belt groove"},
           {"bore", 8, 0, 60, false, "shaft hole diameter"},
           {"hub_diameter", 0, 0, 100, false, "hub diameter (0 = automatic)"},
           {"hub_height", 8, 0, 40, false, "hub height on one side (0 = none)"}}},
         [](Params& p, std::vector<std::string>&) {
             const float D = p["diameter"], W = p["width"], B = std::min(p["bore"], D * 0.5f);
             const float tube = std::min(2 * p["groove_depth"], W - 4);
             const float hub = p["hub_diameter"] > 0 ? std::min(p["hub_diameter"], D - 10) : std::min(std::max(B * 2.2f, 16.f), D - 10);
             Parts s("aluminium");
             s.add("cylinder", {D, W}, 0, W / 2, 0);
             if (p["hub_height"] > 0) s.add("cylinder", {hub, W + p["hub_height"]}, 0, (W + p["hub_height"]) / 2, 0).blend(1.5f);
             s.cut("torus", {D, tube}, 0, W / 2, 0);
             if (B > 0) s.cut("cylinder", {B, W + p["hub_height"] + 10}, 0, (W + p["hub_height"]) / 2, 0);
             s.bore_ports(W + p["hub_height"]);
             return s.json("V-belt pulley");
         }},

        {{"knurled_knob", "Round thumb knob with grip ridges around the edge and a blind hole underneath. Also: knurled, grippy or fluted knob, thumb wheel, dial.",
          {{"diameter", 30, 10, 100, false, "outer diameter"},
           {"height", 16, 6, 60, false, "height"},
           {"ridges", 12, 6, 60, true, "number of grip ridges"},
           {"bore", 6, 0, 30, false, "hole diameter underneath (0 = none)"}}},
         [](Params& p, std::vector<std::string>&) {
             const float D = p["diameter"], H = p["height"], n = p["ridges"];
             const float r = std::min(kPi * D / n * 0.28f, D * 0.12f);  // groove radius
             Parts s("black");
             s.add("cylinder", {D, H * 0.8f}, 0, H * 0.4f, 0);
             s.add("cone", {D, D * 0.82f, H * 0.2f}, 0, H * 0.9f, 0).blend(1);
             s.cut("cylinder", {2 * r, H + 10}, D / 2 + r * 0.35f, H / 2, 0).repeat(int(n));
             if (p["bore"] > 0) s.cut("cylinder", {std::min(p["bore"], D * 0.5f), H * 1.3f}, 0, 0, 0);
             s.port("socket", 0, 0, 0, 0, -1, 0);
             return s.json("knurled knob");
         }},

        {{"star_knob", "Star (lobed) knob with a hex socket underneath for a nut or bolt head. Also: clamping knob, lobed or star grip, wing knob.",
          {{"diameter", 40, 15, 120, false, "overall diameter across the lobes"},
           {"height", 14, 6, 50, false, "height"},
           {"lobes", 5, 3, 10, true, "number of lobes"},
           {"socket", 10, 0, 40, false, "hex socket across flats (0 = none)"}}},
         [](Params& p, std::vector<std::string>&) {
             const float D = p["diameter"], H = p["height"], n = p["lobes"];
             const float lobe = std::min(D * 0.36f, kPi * D / n * 0.55f);
             Parts s("black");
             s.add("cylinder", {D * 0.6f, H}, 0, H / 2, 0);
             s.add("cylinder", {lobe, H}, D / 2 - lobe / 2, H / 2, 0).repeat(int(n)).blend(4);
             s.add("cylinder", {D * 0.45f, H * 0.35f}, 0, H + H * 0.175f - 1, 0).blend(2);
             if (p["socket"] > 0) {
                 const float S = std::min(p["socket"], D * 0.45f);
                 s.cut("prism", {S, H * 1.2f, 6}, 0, 0, 0);
             }
             s.port("socket", 0, 0, 0, 0, -1, 0);
             return s.json("star knob");
         }},

        {{"hex_nut", "ISO hex nut. Give the thread size: M12 -> thread 12. Sizes follow ISO 4032 (threads not modelled). Also: nut, hexagon nut.",
          {{"thread", 12, 2, 48, false, "thread diameter (M size)"},
           {"across_flats", 0, 0, 80, false, "override width across flats (0 = ISO)"},
           {"height", 0, 0, 50, false, "override height (0 = ISO)"}}},
         [](Params& p, std::vector<std::string>&) {
             const float M = p["thread"];
             const float AF = p["across_flats"] > M + 2 ? p["across_flats"] : nut_across_flats(M);
             const float Hn = p["height"] > 1 ? p["height"] : nut_height(M);
             Parts s("steel");
             s.add("prism", {AF, Hn, 6}, 0, Hn / 2, 0);
             s.cut("cylinder", {M, Hn + 10}, 0, Hn / 2, 0);
             s.bore_ports(Hn);
             return s.json("M" + std::to_string(int(M)) + " hex nut");
         }},

        {{"hand_wheel", "Spoked hand wheel: round rim, centre hub with a bore, straight spokes. Also: valve wheel, steering wheel, spoked wheel.",
          {{"diameter", 100, 40, 220, false, "outer diameter"},
           {"spokes", 5, 2, 12, true, "number of spokes"},
           {"rim_thickness", 10, 4, 30, false, "rim tube diameter"},
           {"hub_diameter", 24, 10, 80, false, "hub diameter"},
           {"hub_height", 20, 6, 60, false, "hub height"},
           {"bore", 8, 0, 50, false, "shaft hole diameter"}}},
         [](Params& p, std::vector<std::string>&) {
             const float D = p["diameter"], R = p["rim_thickness"], Hd = std::min(p["hub_diameter"], D * 0.4f);
             const float Hh = std::max(p["hub_height"], R), y = Hh / 2;
             const float inner = Hd * 0.35f, outer = D / 2 - R / 2;
             const float spoke = std::max(R * 0.7f, 4.f);
             Parts s("red");
             s.add("torus", {D - R, R}, 0, y, 0);
             s.add("cylinder", {Hd, Hh}, 0, y, 0);
             s.add("box", {outer - inner, spoke, spoke}, (inner + outer) / 2, y, 0).repeat(int(p["spokes"])).blend(2).round(spoke * 0.3f);
             if (p["bore"] > 0) s.cut("cylinder", {std::min(p["bore"], Hd - 6), Hh + 10}, 0, y, 0);
             s.bore_ports(Hh);
             return s.json("hand wheel");
         }},

        {{"wall_hook", "Wall hook printed flat: a back plate with countersunk screw holes and a hook arm. Also: coat, towel, key, hat or robe hook.",
          {{"width", 30, 15, 80, false, "plate width"},
           {"length", 70, 30, 160, false, "plate length (up the wall)"},
           {"thickness", 6, 3, 15, false, "plate thickness"},
           {"hook_length", 40, 15, 100, false, "how far the hook sticks out"},
           {"hook_diameter", 10, 5, 25, false, "hook arm thickness"},
           {"holes", 2, 0, 4, true, "number of screw holes"},
           {"hole_diameter", 4.5f, 2, 10, false, "screw hole diameter"}}},
         [](Params& p, std::vector<std::string>&) {
             const float W = p["width"], L = p["length"], T = p["thickness"], A = p["hook_length"];
             const float d = std::min(p["hook_diameter"], W - 4), h = p["hole_diameter"];
             const float z_arm = -L / 2 + d / 2 + 4, tip = A * 0.4f;
             Parts s("white");
             s.add("box", {W, T, L}, 0, T / 2, 0).round(std::min(3.f, T / 2.2f));
             s.add("capsule", {d, A + d}, 0, T + A / 2 - d / 2 + 1, z_arm).blend(3);
             s.add("capsule", {d, tip + d}, 0, T + A - d / 2, z_arm + tip / 2).rotate(90, 0, 0);
             const int n = int(p["holes"]);
             for (int i = 0; i < n; ++i) {
                 const float z0 = z_arm + d / 2 + h + 6, z1 = L / 2 - h - 4;
                 const float z = n == 1 ? (z0 + z1) / 2 : z0 + (z1 - z0) * i / (n - 1);
                 s.cut("cylinder", {h, T + 10}, 0, T / 2, z);
                 s.cut("cone", {h, h * 2.1f, h * 0.55f}, 0, T - h * 0.275f + 0.01f, z);  // countersink
                 s.port("hole_" + std::to_string(i + 1), 0, T, z, 0, 1, 0);
             }
             s.port("back", 0, 0, 0, 0, -1, 0);  // the side that goes against the wall
             return s.json("wall hook");
         }},

        {{"cable_organizer", "Desk cable organizer: a block with round cable channels that cables push into from the top. Also: cable holder, cable clip, cord or charger organiser (one slot per cable).",
          {{"length", 80, 30, 200, false, "length"},
           {"depth", 30, 15, 80, false, "depth (front to back)"},
           {"height", 25, 12, 60, false, "height"},
           {"slots", 3, 1, 10, true, "number of cable channels"},
           {"slot_diameter", 10, 3, 25, false, "channel diameter"}}},
         [](Params& p, std::vector<std::string>& notes) {
             float L = p["length"], Dp = p["depth"], H = p["height"], S = p["slot_diameter"];
             const int n = int(p["slots"]);
             if (L < n * (S + 4) + 6) {
                 L = n * (S + 4) + 6;
                 notes.push_back("length increased to fit all channels");
             }
             H = std::max(H, S + 8);
             const float yc = H - S / 2 - 3;
             Parts s("black");
             s.add("box", {L, H, Dp}, 0, H / 2, 0).round(2);
             for (int i = 0; i < n; ++i) {
                 const float x = -L / 2 + L * (i + 0.5f) / n;
                 s.cut("cylinder", {S, Dp + 10}, x, yc, 0).rotate(90, 0, 0);
                 s.cut("box", {S * 0.6f, (H - yc) * 2, Dp + 10}, x, H, 0);  // entry slit
             }
             return s.json("cable organizer");
         }},

        {{"spur_gear", "Spur gear with straight-sided teeth (simplified profile, not involute) and a bore. Also: gear, cog, gear wheel, sprocket-like wheel.",
          {{"teeth", 20, 8, 80, true, "number of teeth"},
           {"module", 2, 0.5f, 5, false, "gear module (pitch diameter = module x teeth)"},
           {"thickness", 8, 3, 40, false, "face width"},
           {"bore", 8, 0, 60, false, "shaft hole diameter"},
           {"hub_diameter", 0, 0, 100, false, "hub diameter (0 = none)"},
           {"hub_height", 6, 0, 30, false, "hub height above the gear"}}},
         [](Params& p, std::vector<std::string>&) {
             const float z = p["teeth"], m = p["module"], T = p["thickness"];
             const float d = m * z, da = d + 2 * m, df = d - 2.5f * m;
             const float tooth_w = kPi * m / 2, tooth_h = (da - df) / 2;
             const float hub = p["hub_diameter"] > 0 ? std::min(p["hub_diameter"], df - 4) : 0;
             const float top = hub ? T + p["hub_height"] : T;
             Parts s("brass");
             s.add("cylinder", {df, T}, 0, T / 2, 0);
             s.add("box", {tooth_h + 0.5f, T, tooth_w}, df / 2 + tooth_h / 2 - 0.25f, T / 2, 0).repeat(int(z)).round(std::min(tooth_w, tooth_h) * 0.2f);
             if (hub) s.add("cylinder", {hub, top}, 0, top / 2, 0).blend(1);
             if (p["bore"] > 0) s.cut("cylinder", {std::min(p["bore"], df - 6), top + 10}, 0, top / 2, 0);
             s.bore_ports(top);
             return s.json(std::to_string(int(z)) + "-tooth spur gear");
         }},

        {{"washer", "Flat washer. Also: spacer ring, shim.",
          {{"outer", 24, 4, 150, false, "outer diameter"},
           {"inner", 13, 1, 140, false, "hole diameter"},
           {"thickness", 2.5f, 0.8f, 20, false, "thickness"}}},
         [](Params& p, std::vector<std::string>&) {
             const float O = p["outer"], I = std::min(p["inner"], O - 3), T = p["thickness"];
             Parts s("steel");
             s.add("cylinder", {O, T}, 0, T / 2, 0);
             s.cut("cylinder", {I, T + 10}, 0, T / 2, 0);
             s.bore_ports(T);
             return s.json("washer");
         }},

        {{"standoff", "Spacer / standoff: a hex (or round) post with a through hole for a screw. Also: PCB standoff, spacer post, pillar.",
          {{"across_flats", 6, 3, 40, false, "hex width across flats (or diameter if sides = 0)"},
           {"length", 20, 3, 120, false, "length"},
           {"hole_diameter", 3.2f, 1, 30, false, "through hole diameter"},
           {"sides", 6, 0, 12, true, "6 = hex, 0 = round"}}},
         [](Params& p, std::vector<std::string>&) {
             const float A = p["across_flats"], L = p["length"], h = std::min(p["hole_diameter"], A - 1.5f);
             Parts s("brass");
             if (p["sides"] >= 3) s.add("prism", {A, L, p["sides"]}, 0, L / 2, 0);
             else s.add("cylinder", {A, L}, 0, L / 2, 0);
             s.cut("cylinder", {h, L + 10}, 0, L / 2, 0);
             s.bore_ports(L);
             return s.json("standoff");
         }},

        {{"l_bracket", "L-shaped angle bracket with screw holes in both legs and a filleted corner. Also: corner bracket, shelf bracket, angle brace.",
          {{"width", 30, 10, 120, false, "bracket width"},
           {"base_length", 50, 15, 200, false, "length of the leg on the ground"},
           {"upright_length", 40, 15, 200, false, "length of the upright leg"},
           {"thickness", 4, 2, 15, false, "plate thickness"},
           {"holes", 2, 0, 4, true, "screw holes per leg"},
           {"hole_diameter", 5, 2, 12, false, "screw hole diameter"}}},
         [](Params& p, std::vector<std::string>&) {
             const float W = p["width"], A = p["base_length"], B = p["upright_length"], T = p["thickness"], h = p["hole_diameter"];
             const int n = int(p["holes"]);
             Parts s("aluminium");
             s.add("box", {W, T, A}, 0, T / 2, 0).round(std::min(1.f, T / 3));
             s.add("box", {W, B, T}, 0, B / 2, -A / 2 + T / 2).blend(std::min(6.f, T * 1.5f)).round(std::min(1.f, T / 3));
             // One column of holes per leg, evenly spaced along the leg.
             for (int i = 0; i < n; ++i) {
                 const float t = n == 1 ? 0.5f : (i + 0.5f) / n;
                 const float zb = -A / 2 + T + 3 + h / 2 + t * (A - T - 6 - h);
                 const float yu = T + 3 + h / 2 + t * (B - T - 6 - h);
                 s.cut("cylinder", {h, T + 10}, 0, T / 2, zb);
                 s.cut("cylinder", {h, T + 10}, 0, yu, -A / 2 + T / 2).rotate(90, 0, 0);
                 const std::string k = std::to_string(i + 1);
                 s.port("base_hole_" + k, 0, T, zb, 0, 1, 0).port("base_hole_" + k + "_under", 0, 0, zb, 0, -1, 0);
                 s.port("upright_hole_" + k, 0, yu, -A / 2 + T, 0, 0, 1).port("upright_hole_" + k + "_back", 0, yu, -A / 2, 0, 0, -1);
             }
             return s.json("L-bracket");
         }},

        {{"bolt", "ISO hex bolt with a real (modelled) thread. Give the size: M8 x 30 -> thread 8, length 30. Coarse pitch and head size follow ISO. Also: screw, hex screw, machine bolt.",
          {{"thread", 8, 3, 24, false, "thread diameter (M size)"},
           {"length", 30, 6, 120, false, "shank length under the head"},
           {"thread_length", 0, 0, 120, false, "threaded length from the tip (0 = all of it)"}}},
         [](Params& p, std::vector<std::string>&) {
             const float M = p["thread"], L = p["length"];
             const float pitch = iso_pitch(M), k = 0.65f * M, AF = nut_across_flats(M);
             const float Lt = p["thread_length"] > 0 ? std::min(p["thread_length"], L) : L;
             Parts s("steel");
             s.add("prism", {AF, k, 6}, 0, k / 2, 0);  // printed head down, shank up
             if (Lt < L) s.add("cylinder", {M, L - Lt + 0.5f}, 0, k + (L - Lt + 0.5f) / 2, 0);
             s.add("thread", {M, Lt, pitch}, 0, k + L - Lt / 2, 0);
             s.port("under_head", 0, k, 0, 0, 1, 0).port("tip", 0, k + L, 0, 0, 1, 0).port("head", 0, 0, 0, 0, -1, 0);
             char name[48];
             std::snprintf(name, sizeof name, "M%g x %g bolt", M, L);
             return s.json(name);
         }},

        {{"shaft", "Round shaft or axle, with an optional D-flat or keyway for a wheel or gear. Also: axle, pin, rod, spindle.",
          {{"diameter", 8, 2, 50, false, "diameter"},
           {"length", 60, 10, 200, false, "length"},
           {"flat", 0, 0, 10, false, "depth of a D-flat along the top half (0 = none)"},
           {"keyway", 0, 0, 14, false, "keyway width along the top half (0 = none)"}}},
         [](Params& p, std::vector<std::string>&) {
             const float D = p["diameter"], L = p["length"], c = std::min(D * 0.1f, 1.f);
             Parts s("steel");
             s.add("cylinder", {D, L - 2 * c}, 0, L / 2, 0);
             s.add("cylinder", {D - 2 * c, L}, 0, L / 2, 0);  // with the line above: chamfered ends
             if (p["flat"] > 0) {
                 const float f = std::min(p["flat"], D * 0.3f);
                 s.cut("box", {D, L / 2 + 1, D}, D - f, L * 0.75f + 0.5f, 0);
             } else if (p["keyway"] > 0) {
                 const float w = std::min(p["keyway"], D * 0.4f);
                 s.cut("box", {w, L / 2 - 4, D * 0.3f}, 0, L * 0.75f, D / 2);
             }
             s.port("top_end", 0, L, 0, 0, 1, 0).port("bottom_end", 0, 0, 0, 0, -1, 0);
             return s.json("shaft");
         }},

        {{"bearing", "Ball bearing as one printed block (outer and inner ring, shields). 608 -> bore 8, outer 22, width 7; 6000 -> 10, 26, 8; 6200 -> 10, 30, 9. Also: ball bearing, skate bearing.",
          {{"bore", 8, 2, 60, false, "inner (bore) diameter"},
           {"outer", 22, 6, 120, false, "outer diameter"},
           {"width", 7, 2, 40, false, "width"}}},
         [](Params& p, std::vector<std::string>& notes) {
             const float B = p["bore"], W = p["width"];
             float O = p["outer"];
             if (O < B + 6) {
                 O = B + 6;
                 notes.push_back("outer increased to leave room for the rings");
             }
             const float ring = std::max((O - B) * 0.18f, 1.2f), recess = std::min(0.6f, W * 0.1f);
             Parts s("steel");
             s.add("cylinder", {O, W}, 0, W / 2, 0);
             s.cut("cylinder", {O - 2 * ring, 2 * recess}, 0, W, 0);  // shields, slightly recessed
             s.cut("cylinder", {O - 2 * ring, 2 * recess}, 0, 0, 0);
             s.add("cylinder", {B + 2 * ring, W}, 0, W / 2, 0);       // inner ring, flush
             s.cut("cylinder", {B, W + 10}, 0, W / 2, 0);
             s.bore_ports(W);
             return s.json("ball bearing");
         }},
    };
    return t;
}

}  // namespace

const std::vector<FamilyInfo>& families() {
    static const std::vector<FamilyInfo> info = [] {
        std::vector<FamilyInfo> v;
        for (const auto& f : table()) v.push_back(f.info);
        return v;
    }();
    return info;
}

namespace {

std::vector<std::string> words_of(const std::string& text) {
    static const std::vector<std::string> stop = {"a", "an", "the", "with", "and", "for", "of", "to", "in", "on", "my",
                                                  "mm", "it", "its", "that", "this", "one", "some", "make", "me", "please",
                                                  "each", "per", "or", "by", "at", "as", "from", "into"};
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&] {
        if (cur.size() > 3 && cur.back() == 's' && cur[cur.size() - 2] != 's') cur.pop_back();  // holes -> hole
        if (cur.size() >= 2 && std::find(stop.begin(), stop.end(), cur) == stop.end()) out.push_back(cur);
        cur.clear();
    };
    for (char c : text) {
        const char l = char(std::tolower(static_cast<unsigned char>(c)));
        if (l >= 'a' && l <= 'z') cur += l;
        else flush();
    }
    flush();
    return out;
}

bool has_m_size(const std::string& text) {  // "M8", "m12 x 30"
    for (size_t i = 0; i + 1 < text.size(); ++i)
        if ((text[i] == 'M' || text[i] == 'm') && std::isdigit(static_cast<unsigned char>(text[i + 1])) &&
            (i == 0 || !std::isalpha(static_cast<unsigned char>(text[i - 1]))))
            return true;
    return false;
}

}  // namespace

std::vector<FamilyScore> rank_families(const std::string& request) {
    const std::vector<std::string> query = words_of(request);
    const bool m_size = has_m_size(request);
    std::vector<FamilyScore> out;
    for (const auto& f : table()) {
        // weight of each word in this family's vocabulary: name and synonyms 3, description 1, params 0.5
        std::map<std::string, float> vocab;
        auto add = [&](const std::string& text, float w) {
            for (const auto& word : words_of(text)) vocab[word] = std::max(vocab[word], w);
        };
        std::string name = f.info.name;
        std::replace(name.begin(), name.end(), '_', ' ');
        add(name, 3);
        const size_t also = f.info.doc.find("Also:");
        add(f.info.doc.substr(0, also), 1);
        if (also != std::string::npos) add(f.info.doc.substr(also + 5), 3);
        for (const auto& p : f.info.params) {
            std::string pn = p.name;
            std::replace(pn.begin(), pn.end(), '_', ' ');
            add(pn, 0.5f);
        }
        float score = 0;
        for (const auto& q : query)
            if (auto it = vocab.find(q); it != vocab.end()) score += it->second;
        if (m_size && std::any_of(f.info.params.begin(), f.info.params.end(), [](const FamilyParam& p) { return p.name == "thread"; }))
            score += 2;
        out.push_back({f.info.name, score});
    }
    std::stable_sort(out.begin(), out.end(), [](const FamilyScore& a, const FamilyScore& b) { return a.score > b.score; });
    return out;
}

std::vector<std::string> select_families(const std::string& request, int k, const std::string& always) {
    const auto ranked = rank_families(request);
    std::vector<std::string> out;
    if (ranked.empty() || ranked[0].score <= 0) {
        for (const auto& r : ranked) out.push_back(r.name);  // nothing matched: offer them all
        return out;
    }
    if (!always.empty() && std::any_of(ranked.begin(), ranked.end(), [&](const FamilyScore& r) { return r.name == always; }))
        out.push_back(always);
    for (const auto& r : ranked)
        if (int(out.size()) < k && r.score > 0 && r.name != always) out.push_back(r.name);
    return out;
}

std::string families_guide(const std::vector<std::string>& names) {
    std::string g =
        "Answer with JSON: {\"family\": <family>, \"params\": {<param>: number, ...}}\n"
        "Leave out params you have no value for (defaults are sensible). Units are millimetres.\n\n";
    for (const auto& n : names)
        for (const auto& f : families()) {
            if (f.name != n) continue;
            g += "- " + f.name + ": " + f.doc + "\n  params: ";
            for (size_t i = 0; i < f.params.size(); ++i) {
                const auto& p = f.params[i];
                char buf[200];
                std::snprintf(buf, sizeof buf, "%s%s (%s, default %g)", i ? "; " : "", p.name.c_str(), p.doc.c_str(), p.value);
                g += buf;
            }
            g += "\n";
        }
    return g;
}

std::string expand_family(const json::Value& root, std::vector<std::string>& notes) {
    const json::Value* f = root.get("family");
    const std::string name = f && f->is_string() ? f->string : "";
    const auto& t = table();
    auto it = std::find_if(t.begin(), t.end(), [&](const Family& fam) { return fam.info.name == name; });
    if (it == t.end()) {
        std::string list;
        for (const auto& fam : t) list += (list.empty() ? "" : ", ") + fam.info.name;
        throw SpecError({"unknown family \"" + name + "\": use one of " + list + ", or \"custom\" with a parts list"});
    }
    Params p;
    for (const auto& fp : it->info.params) p[fp.name] = fp.value;
    if (const json::Value* given = root.get("params"); given && given->is_object()) {
        for (const auto& [key, v] : given->object) {
            auto fp = std::find_if(it->info.params.begin(), it->info.params.end(), [&](const FamilyParam& q) { return q.name == key; });
            if (fp == it->info.params.end()) {
                if (v.is_number() && v.number != 0) notes.push_back("\"" + key + "\" does not apply to " + name + " (ignored)");
                continue;
            }
            if (!v.is_number() || !std::isfinite(v.number)) continue;
            float x = float(v.number);
            if (fp->integer) x = std::round(x);
            const float c = std::clamp(x, fp->lo, fp->hi);
            if (c != x) {
                char buf[160];
                std::snprintf(buf, sizeof buf, "%s: %s %g is outside %g to %g, used %g", name.c_str(), key.c_str(), x, fp->lo, fp->hi, c);
                notes.push_back(buf);
            }
            p[key] = c;
        }
    }
    return it->make(p, notes);
}

}  // namespace vibecad::csg
