// Minimal neural-network inference in plain C++ (no BLAS, no framework): the layers the
// text-to-motion models in src/animation use, with PyTorch's exact conventions so outputs
// match torch to float precision (see src/animation/test_motion_models.py).
//
// Layouts: a sequence of tokens is row-major [n, d]; a 1D signal is channels-first [c, t].
#pragma once
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace vibecad::nn {

using Vec = std::vector<float>;

// y[n, out] = x[n, in] W[out, in]^T + b[out]   (torch.nn.Linear)
void linear(const float* x, int n, int in, const float* W, const float* b, int out, float* y);

// Per row of [n, d]: (x - mean) / sqrt(var + eps) * g + b   (torch.nn.LayerNorm, biased var)
void layer_norm(float* x, int n, int d, const float* g, const float* b, float eps = 1e-5f);

// Per group of channels of [c, t]: normalize over (c / groups) * t values   (torch.nn.GroupNorm)
// With batch > 1 each channel row holds `batch` samples of length t, normalized separately.
void group_norm(float* x, int c, int t, int groups, const float* g, const float* b, float eps = 1e-5f, int batch = 1);

// C[M, N] = A[M, K] B[K, N], row-major (register-blocked, vectorizes without intrinsics)
void gemm(const float* A, const float* B, float* C, int M, int K, int N);

// [cin, t] -> [cout, t_out]; W[cout, cin, k]   (torch.nn.Conv1d). With batch > 1, each row holds
// `batch` samples of length t side by side ([c, batch * t]) and one GEMM serves them all.
Vec conv1d(const Vec& x, int cin, int t, const float* W, const float* b, int cout, int k, int stride, int pad, int batch = 1);
// [cin, t] -> [cout, (t - 1) * stride - 2 pad + k]; W[cin, cout, k]   (torch.nn.ConvTranspose1d)
Vec conv_transpose1d(const Vec& x, int cin, int t, const float* W, const float* b, int cout, int k, int stride, int pad, int batch = 1);

inline float silu(float v) { return v / (1 + std::exp(-v)); }
void silu(float* x, size_t n);
void relu(float* x, size_t n);

// Portable random numbers: the same seed gives the same motion in Python, native code and
// WebAssembly (std::normal_distribution differs between standard libraries).
class Rng {
public:
    explicit Rng(uint64_t seed) : s_(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint64_t next();        // splitmix64
    float uniform();        // (0, 1)
    float normal();         // N(0, 1), Box-Muller
private:
    uint64_t s_;
    bool has_spare_ = false;
    float spare_ = 0;
};

// Named float32 tensors and a JSON header, loaded from a ".vcw" file (see motion_models.hpp).
struct Weights {
    std::string header;  // JSON
    std::vector<float> data;
    struct Entry {
        size_t offset = 0;
        std::vector<int> shape;
        size_t size() const;
    };
    std::vector<std::pair<std::string, Entry>> tensors;

    const float* get(const std::string& name, std::initializer_list<int> shape) const;  // checks the shape
    static Weights parse(const uint8_t* bytes, size_t n);
};

}  // namespace vibecad::nn
