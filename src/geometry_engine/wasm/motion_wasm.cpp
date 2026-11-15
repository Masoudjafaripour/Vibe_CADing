// WebAssembly entry points for the website's live text-to-motion demo: the C++ inference of
// src/animation's AR Transformer and DDPM (motion_models.cpp), writing BVH for the player.
// Build: ./build_motion.sh  (Emscripten)
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <map>
#include <memory>

#include "vibecad/motion_models.hpp"

using namespace emscripten;
using namespace vibecad::motion;

namespace {

std::map<std::string, std::unique_ptr<MotionModel>> models;

val strings(const std::vector<std::string>& v) {
    val out = val::array();
    for (const auto& s : v) out.call<void>("push", s);
    return out;
}

// bytes: the .vcw file as a Uint8Array. Returns {ok, error?, method, words, frames, fps}.
val load_model(const std::string& name, const val& bytes) {
    val out = val::object();
    try {
        const std::vector<uint8_t> data = convertJSArrayToNumberVector<uint8_t>(bytes);
        auto m = MotionModel::load(data.data(), data.size());
        out.set("method", m->method());
        out.set("words", strings(m->vocabulary()));
        out.set("frames", m->frames());
        out.set("fps", m->fps());
        models[name] = std::move(m);
        out.set("ok", true);
    } catch (const std::exception& e) {
        out.set("ok", false);
        out.set("error", std::string(e.what()));
    }
    return out;
}

// Returns {ok, error?, bvh, ms, frames, used: [known words that were used]}.
val generate(const std::string& name, const std::string& text, double seed, int steps) {
    val out = val::object();
    auto it = models.find(name);
    if (it == models.end()) {
        out.set("ok", false), out.set("error", std::string("model not loaded"));
        return out;
    }
    try {
        const MotionModel& m = *it->second;
        const GeneratedClip c = m.generate(text, uint64_t(seed), steps);
        out.set("bvh", bvh_text(m.skeleton(), c.root.data(), c.angles.data(), c.frames, m.fps()));
        out.set("ms", c.ms);
        out.set("frames", c.frames);
        std::vector<std::string> used;
        for (int id : c.words)
            if (id > 0) used.push_back(m.vocabulary()[id - 1]);
        out.set("used", strings(used));
        double travel = 0;  // metres the pelvis moves, for the HUD
        travel = std::hypot(c.root[(c.frames - 1) * 3] - c.root[0], c.root[(c.frames - 1) * 3 + 2] - c.root[2]);
        out.set("travel", travel);
        out.set("ok", true);
    } catch (const std::exception& e) {
        out.set("ok", false);
        out.set("error", std::string(e.what()));
    }
    return out;
}

}  // namespace

EMSCRIPTEN_BINDINGS(vibecad_motion) {
    function("loadModel", &load_model);
    function("generate", &generate);
}
