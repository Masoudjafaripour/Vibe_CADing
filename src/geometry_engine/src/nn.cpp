#include "vibecad/nn.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "vibecad/json.hpp"

namespace vibecad::nn {

void linear(const float* x, int n, int in, const float* W, const float* b, int out, float* y) {
    for (int r = 0; r < n; ++r) {
        const float* xr = x + size_t(r) * in;
        float* yr = y + size_t(r) * out;
        for (int o = 0; o < out; ++o) {
            const float* w = W + size_t(o) * in;
            float s = 0;
            for (int i = 0; i < in; ++i) s += w[i] * xr[i];
            yr[o] = s + (b ? b[o] : 0);
        }
    }
}

void layer_norm(float* x, int n, int d, const float* g, const float* b, float eps) {
    for (int r = 0; r < n; ++r) {
        float* v = x + size_t(r) * d;
        double mean = 0, var = 0;
        for (int i = 0; i < d; ++i) mean += v[i];
        mean /= d;
        for (int i = 0; i < d; ++i) var += (v[i] - mean) * (v[i] - mean);
        const float inv = float(1 / std::sqrt(var / d + eps));
        for (int i = 0; i < d; ++i) v[i] = float(v[i] - mean) * inv * g[i] + b[i];
    }
}

void group_norm(float* x, int c, int t, int groups, const float* g, const float* b, float eps, int batch) {
    const int per = c / groups, row = t * batch;  // each channel row holds `batch` samples of length t
    for (int s = 0; s < batch; ++s)
        for (int gi = 0; gi < groups; ++gi) {
            double mean = 0, var = 0;
            for (int ch = 0; ch < per; ++ch) {
                const float* v = x + size_t(gi * per + ch) * row + size_t(s) * t;
                for (int i = 0; i < t; ++i) mean += v[i];
            }
            const double m = double(per) * t;
            mean /= m;
            for (int ch = 0; ch < per; ++ch) {
                const float* v = x + size_t(gi * per + ch) * row + size_t(s) * t;
                for (int i = 0; i < t; ++i) var += (v[i] - mean) * (v[i] - mean);
            }
            const float inv = float(1 / std::sqrt(var / m + eps));
            for (int ch = 0; ch < per; ++ch) {
                const int cc = gi * per + ch;
                float* v = x + size_t(cc) * row + size_t(s) * t;
                for (int i = 0; i < t; ++i) v[i] = float(v[i] - mean) * inv * g[cc] + b[cc];
            }
        }
}

// C[M, N] = A[M, K] B[K, N]: four rows of C at a time, so every load of a B row feeds four
// multiply-adds (the inner loop over N vectorizes; no BLAS, no intrinsics, same code in WebAssembly).
void gemm(const float* A, const float* B, float* C, int M, int K, int N) {
    constexpr int R = 8, NB = 128;
    float acc[R][NB];
    for (int j0 = 0; j0 < N; j0 += NB) {
        const int nb = std::min(NB, N - j0);
        for (int i = 0; i < M; i += R) {
            const int rows = std::min(R, M - i);
            for (int r = 0; r < rows; ++r) std::fill(acc[r], acc[r] + nb, 0.f);
            if (rows == R) {
                for (int k = 0; k < K; ++k) {
                    const float* b = B + size_t(k) * N + j0;
                    float w[R];
                    for (int r = 0; r < R; ++r) w[r] = A[size_t(i + r) * K + k];
                    for (int j = 0; j < nb; ++j) {
                        const float bj = b[j];
                        for (int r = 0; r < R; ++r) acc[r][j] += w[r] * bj;
                    }
                }
            } else {
                for (int r = 0; r < rows; ++r)
                    for (int k = 0; k < K; ++k) {
                        const float* b = B + size_t(k) * N + j0;
                        const float wr = A[size_t(i + r) * K + k];
                        for (int j = 0; j < nb; ++j) acc[r][j] += wr * b[j];
                    }
            }
            for (int r = 0; r < rows; ++r) std::copy(acc[r], acc[r] + nb, C + size_t(i + r) * N + j0);
        }
    }
}

Vec conv1d(const Vec& x, int cin, int t, const float* W, const float* b, int cout, int k, int stride, int pad, int batch) {
    // im2col: col[(ci, kk), (s, i)] = x_padded[ci, (s, i * stride + kk)]; then out = W[cout, cin * k] col.
    // All samples of the batch share one GEMM, so each weight is loaded once for all of them.
    const int t_out = (t + 2 * pad - k) / stride + 1, n = t_out * batch;
    Vec col(size_t(cin) * k * n, 0.f);
    for (int ci = 0; ci < cin; ++ci)
        for (int kk = 0; kk < k; ++kk) {
            float* row = &col[(size_t(ci) * k + kk) * n];
            for (int s = 0; s < batch; ++s) {
                const float* xs = &x[size_t(ci) * t * batch + size_t(s) * t];
                for (int i = 0; i < t_out; ++i) {
                    const int src = i * stride + kk - pad;
                    if (src >= 0 && src < t) row[s * t_out + i] = xs[src];
                }
            }
        }
    Vec y(size_t(cout) * n);
    gemm(W, col.data(), y.data(), cout, cin * k, n);
    if (b)
        for (int co = 0; co < cout; ++co)
            for (int i = 0; i < n; ++i) y[size_t(co) * n + i] += b[co];
    return y;
}

Vec conv_transpose1d(const Vec& x, int cin, int t, const float* W, const float* b, int cout, int k, int stride, int pad, int batch) {
    // cols[(co, kk), i] = sum_ci W[ci, co, kk] x[ci, i]   (a GEMM with W transposed), then
    // col2im: out[co, i * stride - pad + kk] += cols[(co, kk), i].
    const int t_out = (t - 1) * stride - 2 * pad + k;
    Vec Wt(size_t(cout) * k * cin);
    for (int ci = 0; ci < cin; ++ci)
        for (int co = 0; co < cout; ++co)
            for (int kk = 0; kk < k; ++kk) Wt[(size_t(co) * k + kk) * cin + ci] = W[(size_t(ci) * cout + co) * k + kk];
    const int n = t * batch;
    Vec cols(size_t(cout) * k * n);
    gemm(Wt.data(), x.data(), cols.data(), cout * k, cin, n);
    Vec y(size_t(cout) * t_out * batch);
    for (int co = 0; co < cout; ++co)
        for (int s = 0; s < batch; ++s) {
            float* yr = &y[size_t(co) * t_out * batch + size_t(s) * t_out];
            std::fill(yr, yr + t_out, b ? b[co] : 0.f);
            for (int kk = 0; kk < k; ++kk) {
                const float* c = &cols[(size_t(co) * k + kk) * n + size_t(s) * t];
                for (int i = 0; i < t; ++i) {
                    const int o = i * stride - pad + kk;
                    if (o >= 0 && o < t_out) yr[o] += c[i];
                }
            }
        }
    return y;
}

void silu(float* x, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] = silu(x[i]);
}
void relu(float* x, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] = std::max(x[i], 0.f);
}

uint64_t Rng::next() {
    uint64_t z = (s_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
float Rng::uniform() { return float((double(next() >> 11) + 0.5) / 9007199254740992.0); }
float Rng::normal() {
    if (has_spare_) {
        has_spare_ = false;
        return spare_;
    }
    const double u1 = uniform(), u2 = uniform();
    const double r = std::sqrt(-2 * std::log(u1)), a = 6.283185307179586 * u2;
    spare_ = float(r * std::sin(a));
    has_spare_ = true;
    return float(r * std::cos(a));
}

size_t Weights::Entry::size() const {
    size_t n = 1;
    for (int s : shape) n *= size_t(s);
    return n;
}

const float* Weights::get(const std::string& name, std::initializer_list<int> shape) const {
    for (const auto& [key, e] : tensors) {
        if (key != name) continue;
        if (!std::equal(e.shape.begin(), e.shape.end(), shape.begin(), shape.end())) {
            std::string got;
            for (int s : e.shape) got += (got.empty() ? "" : "x") + std::to_string(s);
            throw std::runtime_error("weights: " + name + " has shape " + got + ", not the expected one");
        }
        return data.data() + e.offset;
    }
    throw std::runtime_error("weights: missing tensor " + name);
}

// File: "VCW1", uint32 header length, JSON header, zero padding to 4 bytes, float32 little-endian data.
// The header's "tensors" maps name -> {"shape": [...], "offset": index of the first float}.
Weights Weights::parse(const uint8_t* bytes, size_t n) {
    if (n < 8 || std::memcmp(bytes, "VCW1", 4) != 0) throw std::runtime_error("weights: not a .vcw file");
    uint32_t len;
    std::memcpy(&len, bytes + 4, 4);
    if (8 + size_t(len) > n) throw std::runtime_error("weights: truncated header");
    Weights w;
    w.header.assign(reinterpret_cast<const char*>(bytes + 8), len);
    const size_t start = (8 + size_t(len) + 3) / 4 * 4;
    w.data.resize((n - start) / 4);
    std::memcpy(w.data.data(), bytes + start, w.data.size() * 4);
    const json::Value root = json::parse(w.header);
    const json::Value* t = root.get("tensors");
    if (!t || !t->is_object()) throw std::runtime_error("weights: header has no tensors");
    for (const auto& [name, v] : t->object) {
        Entry e;
        e.offset = size_t(v.get("offset")->number);
        for (const auto& s : v.get("shape")->array) e.shape.push_back(int(s.number));
        if (e.offset + e.size() > w.data.size()) throw std::runtime_error("weights: " + name + " runs past the end");
        w.tensors.push_back({name, e});
    }
    return w;
}

}  // namespace vibecad::nn
