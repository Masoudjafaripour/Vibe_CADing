// Text -> motion in C++: the two generative models of src/animation, ported for inference.
//
//   "wave hello" -> word ids -> AR Transformer (frame by frame, KV cache)        -> features (T, F)
//                            or DDPM U-Net (100 denoising steps, classifier-free guidance)
//               -> de-normalize -> root + joint angles -> forward kinematics / BVH
//
// Training stays in PyTorch (src/animation/generate_motion.py); export_weights.py writes the
// trained weights to a .vcw file that this code loads. test_motion_models.py checks that every
// layer, an AR rollout and a sampler run with the same noise match PyTorch.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vibecad/motion.hpp"
#include "vibecad/nn.hpp"

namespace vibecad::motion {

struct GeneratedClip {
    std::vector<double> root;    // (T, 3)
    std::vector<double> angles;  // (T, J, 3), XYZ Euler, radians
    std::vector<int> words;      // the word ids used
    int frames = 0;
    double ms = 0;               // model time
};

class MotionModel {
public:
    // Loads a .vcw file exported by src/animation/export_weights.py.
    static std::unique_ptr<MotionModel> load(const uint8_t* bytes, size_t n);
    static std::unique_ptr<MotionModel> load_file(const std::string& path);
    virtual ~MotionModel() = default;

    const std::string& method() const { return method_; }  // "ar" or "diffusion"
    const std::vector<std::string>& vocabulary() const { return words_; }  // known words (id = index + 1)
    const Skeleton& skeleton() const { return skeleton_; }
    int frames() const { return seq_len_; }
    double fps() const { return fps_; }
    int features() const { return n_active_; }

    // Text -> up to 8 word ids (unknown words dropped, PAD = 0). Throws if no word is known.
    std::vector<int> encode(const std::string& text) const;

    // text -> clip. `seed` drives the diffusion noise (the AR model is deterministic);
    // `steps` < 100 samples diffusion on fewer, evenly spaced timesteps (faster, a little rougher).
    GeneratedClip generate(const std::string& text, uint64_t seed = 0, int steps = 0) const;

    // Normalized features (T, F) -> root (T, 3) and angles (T, J, 3), as generate_motion.Generator does.
    void decode(const std::vector<float>& x, std::vector<double>& root, std::vector<double>& angles) const;

    // Model outputs in normalized feature space, for checking against PyTorch.
    virtual std::vector<float> sample(const std::vector<int>& words, uint64_t seed, int steps) const = 0;

protected:
    std::string method_;
    std::vector<std::string> words_;
    std::vector<float> mean_, std_;
    std::vector<int> active_;  // feature indices the model predicts
    int n_features_ = 0, n_active_ = 0, seq_len_ = 90;
    double fps_ = 30;
    Skeleton skeleton_;
    nn::Weights w_;

    // EmbeddingBag(mode="mean", padding_idx=0): mean of the non-PAD rows, zeros if there are none.
    std::vector<float> text_embedding(const std::vector<int>& words, const float* table, int dim) const;
};

// Autoregressive Transformer (model.MotionTransformer): pre-norm encoder layers, causal attention.
class ArModel : public MotionModel {
public:
    std::vector<float> sample(const std::vector<int>& words, uint64_t seed, int steps) const override;
    // Full forward pass, words + frames (T, F) -> outputs (T + 1, F), as MotionTransformer.forward.
    std::vector<float> forward(const std::vector<int>& words, const std::vector<float>& frames, int t) const;

private:
    friend class MotionModel;
    int d_ = 128, layers_ = 3, heads_ = 4, ff_ = 512;
    struct Cache;
    // Runs one token through every layer, appending its keys and values to the cache; returns head(h).
    std::vector<float> step(std::vector<float> h, Cache& cache) const;
    const float* positions(int n) const;
};

// DDPM with a 1D temporal U-Net (diffusion.MotionDenoiser + GaussianDiffusion).
class DiffusionModel : public MotionModel {
public:
    std::vector<float> sample(const std::vector<int>& words, uint64_t seed, int steps) const override;
    // G(x_t, t, words) -> x0_hat, x (T, F) -> (T, F).
    std::vector<float> denoise(const std::vector<float>& x, int t, const std::vector<int>& words) const;
    // Ancestral sampling with the given noise: noise[0] is x_K, noise[k] the draw at step K - k.
    // Exactly GaussianDiffusion.sample, so it can be compared with PyTorch fed the same noise.
    std::vector<float> sample_with_noise(const std::vector<int>& words, const std::vector<std::vector<float>>& noise,
                                         float guidance) const;
    float guidance = 2.0f;
    int timesteps() const { return steps_; }

private:
    friend class MotionModel;
    int ch_ = 128, steps_ = 100;
    std::vector<double> betas_, alphas_, abar_;
    std::vector<float> resblock(const std::vector<float>& x, int cin, int t, const std::string& name, int cout,
                                const std::vector<std::vector<float>>& emb) const;
    // The same x_t with several texts in one pass (classifier-free guidance needs two).
    std::vector<std::vector<float>> denoise_batch(const std::vector<float>& x, int t,
                                                  const std::vector<std::vector<int>>& words) const;
    std::vector<float> sample_impl(const std::vector<int>& words, const std::vector<int>& schedule, float guidance,
                                   const std::function<float()>& noise) const;
};

// BVH text (the file write_bvh writes), for callers without a file system.
std::string bvh_text(const Skeleton& skel, const double* root, const double* angles, int frames, double fps);

}  // namespace vibecad::motion
