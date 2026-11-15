#include "vibecad/motion_models.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "vibecad/json.hpp"

namespace vibecad::motion {
namespace {

constexpr int kPad = 0, kMaxWords = 8;

std::vector<std::string> tokenize(const std::string& text) {  // re.findall(r"[a-z]+", text.lower())
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        const char l = char(std::tolower(static_cast<unsigned char>(c)));
        if (l >= 'a' && l <= 'z') cur += l;
        else if (!cur.empty()) out.push_back(cur), cur.clear();
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::vector<float> numbers(const json::Value* v) {
    std::vector<float> out;
    if (v && v->is_array())
        for (const auto& x : v->array) out.push_back(x.type == json::Value::Type::Bool ? float(x.boolean) : float(x.number));
    return out;
}

int number(const json::Value& root, const char* key, int fallback) {
    const json::Value* v = root.get(key);
    return v && v->is_number() ? int(v->number) : fallback;
}

}  // namespace

// ---------------------------------------------------------------- loading

std::unique_ptr<MotionModel> MotionModel::load(const uint8_t* bytes, size_t n) {
    nn::Weights w = nn::Weights::parse(bytes, n);
    const json::Value h = json::parse(w.header);
    const json::Value* m = h.get("method");
    const std::string method = m && m->is_string() ? m->string : "";
    std::unique_ptr<MotionModel> model;
    if (method == "ar") {
        auto ar = std::make_unique<ArModel>();
        if (const json::Value* c = h.get("config")) {
            ar->d_ = number(*c, "d_model", 128), ar->layers_ = number(*c, "layers", 3), ar->heads_ = number(*c, "heads", 4);
            ar->ff_ = number(*c, "ff", 4 * ar->d_);
        }
        model = std::move(ar);
    } else if (method == "diffusion") {
        auto df = std::make_unique<DiffusionModel>();
        if (const json::Value* c = h.get("config")) df->ch_ = number(*c, "ch", 128), df->steps_ = number(*c, "steps", 100);
        // Cosine schedule (Nichol & Dhariwal 2021), as GaussianDiffusion.__init__.
        const int K = df->steps_;
        std::vector<double> abar(K + 1);
        for (int i = 0; i <= K; ++i) {
            const double s = double(i) / K, c = std::cos((s + 0.008) / 1.008 * 3.14159265358979323846 / 2);
            abar[i] = c * c;
        }
        for (int i = K; i >= 0; --i) abar[i] /= abar[0];
        double prod = 1;
        for (int i = 0; i < K; ++i) {
            const double beta = std::min(1 - abar[i + 1] / abar[i], 0.999);
            df->betas_.push_back(beta), df->alphas_.push_back(1 - beta);
            prod *= 1 - beta;
            df->abar_.push_back(prod);
        }
        model = std::move(df);
    } else {
        throw std::runtime_error("weights: unknown method \"" + method + "\"");
    }
    model->method_ = method;
    for (const auto& word : h.get("words")->array) model->words_.push_back(word.string);
    model->mean_ = numbers(h.get("mean")), model->std_ = numbers(h.get("std"));
    const std::vector<float> active = numbers(h.get("active"));
    for (size_t i = 0; i < active.size(); ++i)
        if (active[i] > 0.5f) model->active_.push_back(int(i));
    model->n_features_ = int(model->mean_.size()), model->n_active_ = int(model->active_.size());
    model->seq_len_ = number(h, "seq_len", 90);
    if (const json::Value* f = h.get("fps"); f && f->is_number()) model->fps_ = f->number;
    const json::Value* sk = h.get("skeleton");
    if (!sk) throw std::runtime_error("weights: no skeleton in the header");
    for (const auto& s : sk->get("names")->array) model->skeleton_.names.push_back(s.string);
    for (const auto& p : sk->get("parents")->array) model->skeleton_.parents.push_back(int(p.number));
    for (const auto& o : sk->get("offsets")->array)
        model->skeleton_.offsets.push_back({o.array[0].number, o.array[1].number, o.array[2].number});
    if (model->n_features_ != 3 + 3 * model->skeleton_.size())
        throw std::runtime_error("weights: feature count does not match the skeleton");
    model->w_ = std::move(w);
    return model;
}

std::unique_ptr<MotionModel> MotionModel::load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return load(bytes.data(), bytes.size());
}

// ---------------------------------------------------------------- shared

std::vector<int> MotionModel::encode(const std::string& text) const {
    std::vector<int> ids;
    for (const std::string& word : tokenize(text)) {
        auto it = std::find(words_.begin(), words_.end(), word);
        if (it != words_.end() && int(ids.size()) < kMaxWords) ids.push_back(int(it - words_.begin()) + 1);
    }
    if (ids.empty()) {
        std::string known;
        for (const auto& w : words_) known += (known.empty() ? "" : ", ") + w;
        throw std::invalid_argument("No known words in \"" + text + "\". Known words: " + known);
    }
    ids.resize(kMaxWords, kPad);
    return ids;
}

std::vector<float> MotionModel::text_embedding(const std::vector<int>& words, const float* table, int dim) const {
    std::vector<float> e(dim, 0.f);
    int n = 0;
    for (int id : words) {
        if (id == kPad) continue;
        const float* row = table + size_t(id) * dim;
        for (int i = 0; i < dim; ++i) e[i] += row[i];
        ++n;
    }
    if (n)
        for (float& v : e) v /= float(n);
    return e;
}

void MotionModel::decode(const std::vector<float>& x, std::vector<double>& root, std::vector<double>& angles) const {
    const int T = int(x.size()) / n_active_, F = n_features_, J = skeleton_.size();
    std::vector<double> feats(size_t(T) * F);
    for (int t = 0; t < T; ++t) {
        for (int f = 0; f < F; ++f) feats[size_t(t) * F + f] = mean_[f];
        for (int a = 0; a < n_active_; ++a) {
            const int f = active_[a];
            feats[size_t(t) * F + f] = double(x[size_t(t) * n_active_ + a]) * std_[f] + mean_[f];
        }
    }
    // from_features: root x/z integrate the velocities, y is the height; then the joint angles.
    root.assign(size_t(T) * 3, 0.0), angles.assign(size_t(T) * J * 3, 0.0);
    double rx = 0, rz = 0;
    for (int t = 0; t < T; ++t) {
        const double* f = &feats[size_t(t) * F];
        rx += f[0] / fps_, rz += f[2] / fps_;
        root[t * 3] = rx, root[t * 3 + 1] = f[1], root[t * 3 + 2] = rz;
        std::copy(f + 3, f + F, angles.begin() + long(t) * J * 3);
    }
}

GeneratedClip MotionModel::generate(const std::string& text, uint64_t seed, int steps) const {
    GeneratedClip clip;
    clip.words = encode(text);
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> x = sample(clip.words, seed, steps);
    clip.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    decode(x, clip.root, clip.angles);
    clip.frames = seq_len_;
    return clip;
}

// ---------------------------------------------------------------- AR Transformer

struct ArModel::Cache {
    std::vector<std::vector<float>> k, v;  // per layer: [n, d]
    int n = 0;
};

// The learned positional table (max_len + 1, d); needs at least `n` rows.
const float* ArModel::positions(int n) const {
    for (const auto& [name, e] : w_.tensors)
        if (name == "pos" && e.shape.size() == 2 && e.shape[1] == d_ && e.shape[0] >= n) return w_.data.data() + e.offset;
    throw std::runtime_error("weights: positional table missing or too short");
}

std::vector<float> ArModel::step(std::vector<float> h, Cache& cache) const {
    const int d = d_, dh = d_ / heads_;
    const float scale = 1 / std::sqrt(float(dh));
    std::vector<float> y(d), qkv(3 * d), attn(d), o(d), ff(ff_);
    for (int l = 0; l < layers_; ++l) {
        const std::string p = "encoder.layers." + std::to_string(l) + ".";
        // x = x + SelfAttention(LayerNorm1(x))   (norm_first)
        y = h;
        nn::layer_norm(y.data(), 1, d, w_.get(p + "norm1.weight", {d}), w_.get(p + "norm1.bias", {d}));
        nn::linear(y.data(), 1, d, w_.get(p + "self_attn.in_proj_weight", {3 * d, d}), w_.get(p + "self_attn.in_proj_bias", {3 * d}),
                   3 * d, qkv.data());
        auto& K = cache.k[l];
        auto& V = cache.v[l];
        K.insert(K.end(), qkv.begin() + d, qkv.begin() + 2 * d);
        V.insert(V.end(), qkv.begin() + 2 * d, qkv.end());
        const int n = cache.n + 1;  // this token sees itself and every earlier one (causal)
        std::vector<float> s(n);
        for (int hd = 0; hd < heads_; ++hd) {
            const float* q = &qkv[size_t(hd) * dh];
            float mx = -1e30f;
            for (int j = 0; j < n; ++j) {
                const float* kj = &K[size_t(j) * d + size_t(hd) * dh];
                float dot = 0;
                for (int i = 0; i < dh; ++i) dot += q[i] * kj[i];
                s[j] = dot * scale;
                mx = std::max(mx, s[j]);
            }
            float sum = 0;
            for (int j = 0; j < n; ++j) sum += (s[j] = std::exp(s[j] - mx));
            float* out = &attn[size_t(hd) * dh];
            std::fill(out, out + dh, 0.f);
            for (int j = 0; j < n; ++j) {
                const float pj = s[j] / sum;
                const float* vj = &V[size_t(j) * d + size_t(hd) * dh];
                for (int i = 0; i < dh; ++i) out[i] += pj * vj[i];
            }
        }
        nn::linear(attn.data(), 1, d, w_.get(p + "self_attn.out_proj.weight", {d, d}), w_.get(p + "self_attn.out_proj.bias", {d}),
                   d, o.data());
        for (int i = 0; i < d; ++i) h[i] += o[i];
        // x = x + Linear2(ReLU(Linear1(LayerNorm2(x))))
        y = h;
        nn::layer_norm(y.data(), 1, d, w_.get(p + "norm2.weight", {d}), w_.get(p + "norm2.bias", {d}));
        nn::linear(y.data(), 1, d, w_.get(p + "linear1.weight", {ff_, d}), w_.get(p + "linear1.bias", {ff_}), ff_, ff.data());
        nn::relu(ff.data(), ff.size());
        nn::linear(ff.data(), 1, ff_, w_.get(p + "linear2.weight", {d, ff_}), w_.get(p + "linear2.bias", {d}), d, o.data());
        for (int i = 0; i < d; ++i) h[i] += o[i];
    }
    ++cache.n;
    std::vector<float> out(n_active_);
    nn::linear(h.data(), 1, d, w_.get("head.weight", {n_active_, d}), w_.get("head.bias", {n_active_}), n_active_, out.data());
    return out;
}

std::vector<float> ArModel::forward(const std::vector<int>& words, const std::vector<float>& frames, int t) const {
    // With causal attention, running the tokens one by one through the KV cache gives exactly
    // the outputs of the full (T + 1)-token pass.
    Cache cache;
    cache.k.resize(layers_), cache.v.resize(layers_);
    const float* pos = positions(t + 1);
    std::vector<float> out;
    for (int i = 0; i <= t; ++i) {
        std::vector<float> h(d_);
        if (i == 0) {
            h = text_embedding(words, w_.get("text.weight", {int(words_.size()) + 1, d_}), d_);
        } else {
            nn::linear(&frames[size_t(i - 1) * n_active_], 1, n_active_, w_.get("inp.weight", {d_, n_active_}),
                       w_.get("inp.bias", {d_}), d_, h.data());
        }
        for (int k = 0; k < d_; ++k) h[k] += pos[size_t(i) * d_ + k];
        const std::vector<float> o = step(h, cache);
        out.insert(out.end(), o.begin(), o.end());
    }
    return out;
}

std::vector<float> ArModel::sample(const std::vector<int>& words, uint64_t, int) const {
    // generate(): roll out seq_len frames, feeding each prediction back in as the next token.
    Cache cache;
    cache.k.resize(layers_), cache.v.resize(layers_);
    const float* pos = positions(seq_len_);
    std::vector<float> h = text_embedding(words, w_.get("text.weight", {int(words_.size()) + 1, d_}), d_);
    std::vector<float> frames;
    for (int i = 0; i < seq_len_; ++i) {
        for (int k = 0; k < d_; ++k) h[k] += pos[size_t(i) * d_ + k];
        const std::vector<float> frame = step(h, cache);
        frames.insert(frames.end(), frame.begin(), frame.end());
        nn::linear(frame.data(), 1, n_active_, w_.get("inp.weight", {d_, n_active_}), w_.get("inp.bias", {d_}), d_, h.data());
    }
    return frames;
}

// ---------------------------------------------------------------- diffusion

// Activations are channels-first with the batch side by side: [c, batch * t].
std::vector<float> DiffusionModel::resblock(const std::vector<float>& x, int cin, int t, const std::string& p, int cout,
                                            const std::vector<std::vector<float>>& emb) const {
    // h = conv1(silu(norm1(x))); h = norm2(h) * (1 + scale) + shift; out = skip(x) + conv2(silu(h))
    const int B = int(emb.size()), n = B * t;
    std::vector<float> a = x;
    nn::group_norm(a.data(), cin, t, 8, w_.get(p + "norm1.weight", {cin}), w_.get(p + "norm1.bias", {cin}), 1e-5f, B);
    nn::silu(a.data(), a.size());
    std::vector<float> h = nn::conv1d(a, cin, t, w_.get(p + "conv1.weight", {cout, cin, 3}), w_.get(p + "conv1.bias", {cout}), cout, 3, 1, 1, B);
    nn::group_norm(h.data(), cout, t, 8, w_.get(p + "norm2.weight", {cout}), w_.get(p + "norm2.bias", {cout}), 1e-5f, B);
    std::vector<float> film(2 * cout);
    for (int s = 0; s < B; ++s) {  // FiLM: per-sample scale and shift from (t, text)
        nn::linear(emb[s].data(), 1, ch_, w_.get(p + "emb.weight", {2 * cout, ch_}), w_.get(p + "emb.bias", {2 * cout}), 2 * cout, film.data());
        for (int c = 0; c < cout; ++c) {
            const float sc = 1 + film[c], sh = film[cout + c];
            float* row = &h[size_t(c) * n + size_t(s) * t];
            for (int i = 0; i < t; ++i) row[i] = row[i] * sc + sh;
        }
    }
    nn::silu(h.data(), h.size());
    std::vector<float> out = nn::conv1d(h, cout, t, w_.get(p + "conv2.weight", {cout, cout, 3}), w_.get(p + "conv2.bias", {cout}), cout, 3, 1, 1, B);
    if (cin != cout) {
        const std::vector<float> sk = nn::conv1d(x, cin, t, w_.get(p + "skip.weight", {cout, cin, 1}), w_.get(p + "skip.bias", {cout}), cout, 1, 1, 0, B);
        for (size_t i = 0; i < out.size(); ++i) out[i] += sk[i];
    } else {
        for (size_t i = 0; i < out.size(); ++i) out[i] += x[i];
    }
    return out;
}

std::vector<std::vector<float>> DiffusionModel::denoise_batch(const std::vector<float>& x, int t,
                                                              const std::vector<std::vector<int>>& words) const {
    const int T = int(x.size()) / n_active_, F = n_active_, C = ch_, B = int(words.size());
    // emb = MLP(sinusoidal(t)) + text embedding, one per sample
    std::vector<float> te(C), e1(C), base(C);
    const int half = C / 2;
    for (int i = 0; i < half; ++i) {
        const float a = float(t) * std::exp(-std::log(10000.f) * float(i) / float(half));
        te[i] = std::sin(a), te[half + i] = std::cos(a);
    }
    nn::linear(te.data(), 1, C, w_.get("time.0.weight", {C, C}), w_.get("time.0.bias", {C}), C, e1.data());
    nn::silu(e1.data(), e1.size());
    nn::linear(e1.data(), 1, C, w_.get("time.2.weight", {C, C}), w_.get("time.2.bias", {C}), C, base.data());
    std::vector<std::vector<float>> emb;
    for (const auto& w : words) {
        std::vector<float> e = text_embedding(w, w_.get("text.weight", {int(words_.size()) + 1, C}), C);
        for (int i = 0; i < C; ++i) e[i] += base[i];
        emb.push_back(e);
    }
    std::vector<float> xc(size_t(F) * B * T);  // (T, F) -> channels-first, the same x for every sample
    for (int s = 0; s < B; ++s)
        for (int i = 0; i < T; ++i)
            for (int f = 0; f < F; ++f) xc[size_t(f) * B * T + size_t(s) * T + i] = x[size_t(i) * F + f];
    const std::vector<float> h0 = nn::conv1d(xc, F, T, w_.get("inp.weight", {C, F, 3}), w_.get("inp.bias", {C}), C, 3, 1, 1, B);
    const std::vector<float> h = resblock(h0, C, T, "enc.", C, emb);
    const int T2 = (T + 2 - 4) / 2 + 1;
    std::vector<float> m = nn::conv1d(h, C, T, w_.get("down.weight", {2 * C, C, 4}), w_.get("down.bias", {2 * C}), 2 * C, 4, 2, 1, B);
    m = resblock(m, 2 * C, T2, "mid.", 2 * C, emb);
    m = nn::conv_transpose1d(m, 2 * C, T2, w_.get("up.weight", {2 * C, C, 4}), w_.get("up.bias", {C}), C, 4, 2, 1, B);
    std::vector<float> cat(h);  // torch.cat([h, m], 1): channels of h, then of m
    cat.insert(cat.end(), m.begin(), m.end());
    const std::vector<float> dcd = resblock(cat, 2 * C, T, "dec.", C, emb);
    const std::vector<float> outc = nn::conv1d(dcd, C, T, w_.get("out.weight", {F, C, 3}), w_.get("out.bias", {F}), F, 3, 1, 1, B);
    std::vector<std::vector<float>> out(B, std::vector<float>(size_t(T) * F));
    for (int s = 0; s < B; ++s)
        for (int i = 0; i < T; ++i)
            for (int f = 0; f < F; ++f) out[s][size_t(i) * F + f] = outc[size_t(f) * B * T + size_t(s) * T + i];
    return out;
}

std::vector<float> DiffusionModel::denoise(const std::vector<float>& x, int t, const std::vector<int>& words) const {
    return denoise_batch(x, t, {words})[0];
}

std::vector<float> DiffusionModel::sample_impl(const std::vector<int>& words, const std::vector<int>& schedule, float g,
                                               const std::function<float()>& noise) const {
    const size_t n = size_t(seq_len_) * n_active_;
    const std::vector<int> null(words.size(), kPad);
    std::vector<float> x(n);
    for (float& v : x) v = noise();
    for (size_t s = 0; s < schedule.size(); ++s) {
        const int i = schedule[s];
        // classifier-free guidance: x0 = G(x, t, 0) + w (G(x, t, c) - G(x, t, 0))
        const auto both = denoise_batch(x, i, {words, null});  // one batched pass for both branches
        const std::vector<float>& c = both[0];
        const std::vector<float>& u = both[1];
        std::vector<float> x0(n);
        for (size_t k = 0; k < n; ++k) x0[k] = u[k] + g * (c[k] - u[k]);
        if (s + 1 == schedule.size()) return x0;
        // posterior q(x_prev | x_t, x0_hat) between this timestep and the next one in the schedule
        const int prev = schedule[s + 1];
        const double abar = abar_[i], abar_prev = abar_[prev];
        const double alpha = prev == i - 1 ? alphas_[i] : abar / abar_prev;
        const double beta = prev == i - 1 ? betas_[i] : 1 - alpha;
        const double c0 = std::sqrt(abar_prev) * beta / (1 - abar), ct = std::sqrt(alpha) * (1 - abar_prev) / (1 - abar);
        const float sd = float(std::sqrt(beta * (1 - abar_prev) / (1 - abar)));
        for (size_t k = 0; k < n; ++k) x[k] = float(c0 * x0[k] + ct * x[k]);
        for (size_t k = 0; k < n; ++k) x[k] += sd * noise();
    }
    return x;
}

std::vector<float> DiffusionModel::sample(const std::vector<int>& words, uint64_t seed, int steps) const {
    // steps < K: an evenly spaced subset of the timesteps, always ending at 0 (respaced sampling).
    const int k = steps > 0 && steps < steps_ ? steps : steps_;
    std::vector<int> schedule;
    for (int s = 0; s < k; ++s) schedule.push_back(int(std::lround(double(steps_ - 1) * (k - 1 - s) / std::max(k - 1, 1))));
    schedule.erase(std::unique(schedule.begin(), schedule.end()), schedule.end());
    nn::Rng rng(seed);
    return sample_impl(words, schedule, guidance, [&] { return rng.normal(); });
}

std::vector<float> DiffusionModel::sample_with_noise(const std::vector<int>& words, const std::vector<std::vector<float>>& noise,
                                                     float g) const {
    std::vector<int> schedule;
    for (int i = steps_ - 1; i >= 0; --i) schedule.push_back(i);
    size_t draw = 0, idx = 0;
    const size_t n = size_t(seq_len_) * n_active_;
    return sample_impl(words, schedule, g, [&]() -> float {
        if (idx == n) idx = 0, ++draw;
        if (draw >= noise.size() || noise[draw].size() != n) throw std::invalid_argument("not enough noise");
        return noise[draw][idx++];
    });
}

}  // namespace vibecad::motion
