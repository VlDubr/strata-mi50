// src/dsv41/glm5_vision.cuh - GLM-5.3-Flash's vision tower (mmproj, projector glm5v; llama.cpp tools/mtmd/models/glm4v.cpp).
//   patches 14 x 14 in 2 x 2 blocks, the two conv kernels summed (one frame)
//   24 blocks: x += attn(rms(x)) (16 heads of 64, q and k each RMS-normed, a 2D rotary: pairs (i, i + 32), i < 16 the row, the rest
//              the column); x += down(swiglu_clamp(gate, up)) with biases
//   rms, each 2 x 2 block as one 4096 row through the merger, fc, LayerNorm, GELU, a clamped SwiGLU of 10240: one row of the
//   model's width a block, in reading order.  The model's attention has no positions, so the rows only replace the image tokens.
// The weights live in pinned host memory as F16 and go to the GPU a block at a time for each picture (about 0.1 s for all of them),
// so the tower costs the card its work buffers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::ds4 {

struct Glm5VisionGeo {
    int layers = 0, dim = 1024, heads = 16, ff = 4096, patch = 14, merge = 2, out = 4096, proj_ff = 10240, max_tokens = 4096;
    float theta = 10000.0f, eps = 1e-5f, clamp = 10.0f;
    float mean[3] = {0, 0, 0}, std[3] = {1, 1, 1};
};

class Glm5Vision {
public:
    /// read(name, n, dst): tensor `name`'s n values as F32; false: it is not there, or has another size
    using Reader = std::function<bool(const std::string& name, size_t n, float* dst)>;
    Glm5Vision() = default;
    Glm5Vision(const Glm5Vision&) = delete;
    Glm5Vision& operator=(const Glm5Vision&) = delete;
    ~Glm5Vision();
    bool load(const Glm5VisionGeo& g, int device, const Reader& read, std::string& err);
    int device() const { return dev_; }
    size_t device_bytes() const { return dev_bytes_; }
    int max_patches() const { return max_n_; }
    /// rgb: height x width x 3 bytes, both sides multiples of patch * merge -> BF16 rows [blocks][out].
    bool encode(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& rows, std::string& err);

private:
    struct Block {
        size_t qkv = 0, o = 0, gate = 0, up = 0, down = 0;   // in host_ (F16 values)
        float *ln1 = nullptr, *ln2 = nullptr, *qn = nullptr, *kn = nullptr, *qkv_b = nullptr, *o_b = nullptr, *gate_b = nullptr,
              *up_b = nullptr, *down_b = nullptr;
    };
    bool dalloc_bytes(void*& p, size_t bytes, std::string& err);
    template <class T>
    bool dalloc(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (!dalloc_bytes(v, n * sizeof(T), err)) return false;
        p = (T*) v;
        return true;
    }
    bool upload_f32(float*& dst, const std::vector<float>& v, std::string& err);
    bool gemm(const void* W, const void* x, float* y, int rows, int in, int out, int ldy, std::string& err);
    bool fetch(size_t off, size_t n, void* dst, std::string& err);

    Glm5VisionGeo g_;
    int dev_ = -1, max_n_ = 0, rows_ff_ = 0, rows_pf_ = 0;
    size_t dev_bytes_ = 0;
    std::vector<void*> allocs_;
    uint16_t* host_ = nullptr;
    std::vector<Block> blocks_;
    size_t m_merge_ = 0, m_fc_ = 0, m_gate_ = 0, m_up_ = 0, m_down_ = 0, slot_n_ = 0;
    uint16_t *patch_w_ = nullptr, *slot_ = nullptr, *xn_ = nullptr, *o_ = nullptr, *mid_ = nullptr, *pt_ = nullptr;
    float *patch_b_ = nullptr, *post_ln_ = nullptr, *merge_b_ = nullptr, *pn_w_ = nullptr, *pn_b_ = nullptr;
    float *x_ = nullptr, *y_ = nullptr, *qkv_ = nullptr, *gb_ = nullptr, *ub_ = nullptr, *mean_ = nullptr, *std_ = nullptr;
    uint8_t* rgb_ = nullptr;
    void* blas_ = nullptr;
    void* st_ = nullptr;
};

}  // namespace strata::ds4
