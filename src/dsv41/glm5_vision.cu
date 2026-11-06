// src/dsv41/glm5_vision.cu - see glm5_vision.cuh.
#include "glm5_vision.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace strata::ds4 {
namespace {

#define VK(call)                                                                          \
    do {                                                                                  \
        cudaError_t e_ = (call);                                                          \
        if (e_ != cudaSuccess) {                                                          \
            err = std::string("glm5 vision: ") + #call + ": " + cudaGetErrorString(e_);   \
            return false;                                                                 \
        }                                                                                 \
    } while (0)

__device__ __forceinline__ uint16_t f2h(float v) { return __half_as_ushort(__float2half(v)); }

// the block's sum over threads (a multiple of 32, at most 1024); every thread gets it
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (lane == 0) part[wid] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += part[i];
    return t;
}

// grid n patches: its pixels as a row, c, ky, kx.  Patches in 2 x 2 blocks, the blocks in reading order
__global__ void patchify_kernel(const uint8_t* __restrict__ rgb, int width, int bw, int patch, const float* __restrict__ mean,
                                const float* __restrict__ sd, uint16_t* __restrict__ out) {
    const int p = blockIdx.x, tok = p >> 2, ty = tok / bw, tx = tok % bw, dy = (p >> 1) & 1, dx = p & 1;
    const int y0 = (ty * 2 + dy) * patch, x0 = (tx * 2 + dx) * patch, pp = patch * patch, n = 3 * pp;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int c = i / pp, r = i % pp, ky = r / patch, kx = r % patch;
        const float v = (float) rgb[((size_t) (y0 + ky) * width + x0 + kx) * 3 + c] * (1.0f / 255.0f);
        out[(size_t) p * n + i] = f2h((v - mean[c]) / sd[c]);
    }
}

__global__ void add_bias_kernel(float* x, const float* __restrict__ b, int cols) {
    const size_t r = blockIdx.x;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) x[r * cols + i] += b[i];
}
__global__ void add_res_kernel(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ b, int cols) {
    const size_t r = blockIdx.x;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) x[r * cols + i] += y[r * cols + i] + b[i];
}
__global__ void rms_f16_kernel(const float* __restrict__ x, const float* __restrict__ w, int n, float eps, uint16_t* __restrict__ y) {
    const float* r = x + (size_t) blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += r[i] * r[i];
    s = block_sum(s);
    const float inv = rsqrtf(s / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[(size_t) blockIdx.x * n + i] = f2h(r[i] * inv * w[i]);
}
__global__ void to_f16_kernel(const float* __restrict__ x, const float* __restrict__ b, int cols, uint16_t* __restrict__ y) {
    const size_t r = blockIdx.x;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) y[r * cols + i] = f2h(x[r * cols + i] + (b != nullptr ? b[i] : 0.0f));
}
__global__ void to_bf16_kernel(const float* __restrict__ x, int cols, uint16_t* __restrict__ y) {
    const size_t r = blockIdx.x;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const uint32_t u = __float_as_uint(x[r * cols + i]);
        y[r * cols + i] = (uint16_t) ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    }
}
// grid (patches, 3), 1024 threads (heads x 64): the bias; q and k normed a head at a time and rotated
__global__ void __launch_bounds__(1024) qk_prep_kernel(float* qkv, const float* __restrict__ bias, const float* __restrict__ qn,
                                                       const float* __restrict__ kn, int bw, float theta, float eps) {
    __shared__ float sh[1024], sq[1024];
    const int p = blockIdx.x, part = blockIdx.y, t = threadIdx.x, h = t >> 6, d = t & 63;
    float* row = qkv + (size_t) p * 3072 + (size_t) part * 1024;
    float v = row[t] + bias[part * 1024 + t];
    if (part == 2) {
        row[t] = v;
        return;
    }
    sq[t] = v * v;
    __syncthreads();
    for (int o = 32; o > 0; o >>= 1) {
        if (d < o) sq[t] += sq[t + o];
        __syncthreads();
    }
    v *= rsqrtf(sq[h * 64] / 64.0f + eps) * (part == 0 ? qn : kn)[d];
    sh[t] = v;
    __syncthreads();
    const int tok = p >> 2, ty = tok / bw, tx = tok % bw, y = ty * 2 + ((p >> 1) & 1), x = tx * 2 + (p & 1);
    const int i = d & 31;
    const float pos = (float) (i < 16 ? y : x);
    const float ang = pos * powf(theta, -(float) (i & 15) / 16.0f);
    const float c = cosf(ang), s = sinf(ang), a = sh[h * 64 + i], b = sh[h * 64 + i + 32];
    row[t] = d < 32 ? a * c - b * s : a * s + b * c;
}
// the bidirectional attention: grid (ceil(n / 64), heads), 256 threads: four threads a query, each its quarter of a tile's keys,
// the four merged at the end.  Exponents in base 2 (scale holds log2 e).
__global__ void __launch_bounds__(256) attn_kernel(const float* __restrict__ qkv, int n, float scale, uint16_t* __restrict__ out) {
    __shared__ float ks[64][68], vs[64][68];
    const int h = blockIdx.y, q0 = blockIdx.x * 64, tid = threadIdx.x, r = tid & 3, row = q0 + (tid >> 2);
    float q[64], acc[64];
    const float* qp = qkv + (size_t) min(row, n - 1) * 3072 + h * 64;
#pragma unroll
    for (int d = 0; d < 64; ++d) {
        q[d] = qp[d] * scale;
        acc[d] = 0.0f;
    }
    float m = -FLT_MAX, l = 0.0f;
    for (int kt = 0; kt < n; kt += 64) {
        __syncthreads();
        for (int i = tid; i < 64 * 16; i += 256) {
            const int j = i >> 4, c = (i & 15) * 4;
            const float* kp = qkv + (size_t) min(kt + j, n - 1) * 3072 + 1024 + h * 64 + c;
            *(float4*) &ks[j][c] = *(const float4*) kp;
            *(float4*) &vs[j][c] = *(const float4*) (kp + 1024);
        }
        __syncthreads();
        float sc[16], mx = -FLT_MAX;
#pragma unroll
        for (int a = 0; a < 16; ++a) {
            const int j = a * 4 + r;
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < 64; d += 4) {
                const float4 k4 = *(const float4*) &ks[j][d];
                s = fmaf(q[d], k4.x, s);
                s = fmaf(q[d + 1], k4.y, s);
                s = fmaf(q[d + 2], k4.z, s);
                s = fmaf(q[d + 3], k4.w, s);
            }
            sc[a] = kt + j < n ? s : -FLT_MAX;
            mx = fmaxf(mx, sc[a]);
        }
        const float mn = fmaxf(m, mx), cr = exp2f(m - mn);
        m = mn;
        l *= cr;
#pragma unroll
        for (int d = 0; d < 64; ++d) acc[d] *= cr;
#pragma unroll
        for (int a = 0; a < 16; ++a) {
            const int j = a * 4 + r;
            const float p = sc[a] == -FLT_MAX ? 0.0f : exp2f(sc[a] - mn);
            l += p;
#pragma unroll
            for (int d = 0; d < 64; d += 4) {
                const float4 v4 = *(const float4*) &vs[j][d];
                acc[d] = fmaf(p, v4.x, acc[d]);
                acc[d + 1] = fmaf(p, v4.y, acc[d + 1]);
                acc[d + 2] = fmaf(p, v4.z, acc[d + 2]);
                acc[d + 3] = fmaf(p, v4.w, acc[d + 3]);
            }
        }
    }
    float M = m;
    M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 1));
    M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 2));
    const float c = exp2f(m - M);
    l *= c;
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    const float inv = 1.0f / l;
#pragma unroll
    for (int d = 0; d < 64; ++d) {
        float v = acc[d] * c;
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        if ((d >> 4) == r && row < n) out[(size_t) row * 1024 + h * 64 + d] = f2h(v * inv);
    }
}
__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, const float* __restrict__ gb,
                              const float* __restrict__ ub, int cols, float lim, uint16_t* __restrict__ y) {
    const size_t r = blockIdx.x;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float a = fminf(g[r * cols + i] + (gb != nullptr ? gb[i] : 0.0f), lim);
        const float b = fminf(fmaxf(u[r * cols + i] + (ub != nullptr ? ub[i] : 0.0f), -lim), lim);
        y[r * cols + i] = f2h(a / (1.0f + expf(-a)) * b);
    }
}
__global__ void ln_gelu_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, int n, float eps,
                               uint16_t* __restrict__ y) {
    const float* r = x + (size_t) blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += r[i];
    const float mean = block_sum(s) / n;
    s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += (r[i] - mean) * (r[i] - mean);
    const float inv = rsqrtf(block_sum(s) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = (r[i] - mean) * inv * w[i] + b[i];
        y[(size_t) blockIdx.x * n + i] = f2h(0.5f * v * (1.0f + erff(v * 0.70710678f)));
    }
}

uint16_t f32_to_f16(float f) { return __half_as_ushort(__float2half(f)); }

}  // namespace

Glm5Vision::~Glm5Vision() {
    if (dev_ >= 0) cudaSetDevice(dev_);
    if (blas_ != nullptr) cublasDestroy((cublasHandle_t) blas_);
    if (st_ != nullptr) cudaStreamDestroy((cudaStream_t) st_);
    for (void* p : allocs_) cudaFree(p);
    if (host_ != nullptr) cudaFreeHost(host_);
}

bool Glm5Vision::dalloc_bytes(void*& p, size_t bytes, std::string& err) {
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        err = "glm5 vision: " + std::to_string(bytes >> 20) + " MiB of graphics memory";
        return false;
    }
    allocs_.push_back(p);
    dev_bytes_ += bytes;
    return true;
}
bool Glm5Vision::upload_f32(float*& dst, const std::vector<float>& v, std::string& err) {
    if (!dalloc(dst, v.size(), err)) return false;
    VK(cudaMemcpy(dst, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    return true;
}

bool Glm5Vision::load(const Glm5VisionGeo& g, int device, const Reader& read, std::string& err) {
    g_ = g;
    dev_ = device;
    VK(cudaSetDevice(device));
    const size_t D = (size_t) g.dim, F = (size_t) g.ff, O = (size_t) g.out, PF = (size_t) g.proj_ff, MR = (size_t) g.merge * g.merge * D;
    if (g.dim != 1024 || g.heads != 16 || g.merge != 2) {
        err = "glm5 vision: a tower of another size";
        return false;
    }
    const size_t per_block = D * 3 * D + D * D + 3 * D * F;
    const size_t total = per_block * (size_t) g.layers + MR * O + O * O + 2 * O * PF + PF * O;
    VK(cudaHostAlloc((void**) &host_, total * 2, cudaHostAllocDefault));
    size_t at = 0;
    std::vector<float> f;
    auto f16 = [&](const std::string& name, size_t n, size_t& off) {
        f.resize(n);
        if (!read(name, n, f.data())) {
            err = "glm5 vision: " + name + " is not there, or of another size";
            return false;
        }
        off = at;
        for (size_t i = 0; i < n; ++i) host_[at + i] = f32_to_f16(f[i]);
        at += n;
        return true;
    };
    auto small = [&](const std::string& name, size_t n, float*& dst) {
        f.resize(n);
        if (!read(name, n, f.data())) {
            err = "glm5 vision: " + name + " is not there, or of another size";
            return false;
        }
        return upload_f32(dst, f, err);
    };
    blocks_.resize((size_t) g.layers);
    for (int b = 0; b < g.layers; ++b) {
        Block& k = blocks_[(size_t) b];
        const std::string P = "vision.v.blk." + std::to_string(b) + ".";
        if (!f16(P + "attn_qkv.weight", D * 3 * D, k.qkv) || !f16(P + "attn_out.weight", D * D, k.o) ||
            !f16(P + "ffn_gate.weight", D * F, k.gate) || !f16(P + "ffn_up.weight", D * F, k.up) ||
            !f16(P + "ffn_down.weight", D * F, k.down) || !small(P + "ln1.weight", D, k.ln1) || !small(P + "ln2.weight", D, k.ln2) ||
            !small(P + "attn_q_norm.weight", 64, k.qn) || !small(P + "attn_k_norm.weight", 64, k.kn) ||
            !small(P + "attn_qkv.bias", 3 * D, k.qkv_b) || !small(P + "attn_out.bias", D, k.o_b) ||
            !small(P + "ffn_gate.bias", F, k.gate_b) || !small(P + "ffn_up.bias", F, k.up_b) || !small(P + "ffn_down.bias", D, k.down_b))
            return false;
    }
    // the merger: the file's [out][c][ky][kx] as [out][ky][kx][c], so a block's four rows are its input row
    {
        std::vector<float> w(MR * O);
        if (!read("vision.mm.patch_merger.weight", w.size(), w.data())) {
            err = "glm5 vision: mm.patch_merger.weight";
            return false;
        }
        m_merge_ = at;
        for (size_t o = 0; o < O; ++o)
            for (size_t c = 0; c < D; ++c)
                for (size_t k = 0; k < 4; ++k) host_[at + o * MR + k * D + c] = f32_to_f16(w[o * MR + c * 4 + k]);
        at += MR * O;
    }
    if (!f16("vision.mm.model.fc.weight", O * O, m_fc_) || !f16("vision.mm.gate.weight", O * PF, m_gate_) ||
        !f16("vision.mm.up.weight", O * PF, m_up_) || !f16("vision.mm.down.weight", PF * O, m_down_) ||
        !small("vision.mm.patch_merger.bias", O, merge_b_) || !small("vision.mm.post_norm.weight", O, pn_w_) ||
        !small("vision.mm.post_norm.bias", O, pn_b_) || !small("vision.v.post_ln.weight", D, post_ln_) ||
        !small("vision.v.patch_embd.bias", D, patch_b_))
        return false;
    // the two conv kernels of a frame pair, summed: a still picture is both frames
    const size_t pe = (size_t) 3 * g.patch * g.patch;
    {
        std::vector<float> a(pe * D), b(pe * D);
        if (!read("vision.v.patch_embd.weight", a.size(), a.data()) || !read("vision.v.patch_embd.weight.1", b.size(), b.data())) {
            err = "glm5 vision: v.patch_embd.weight";
            return false;
        }
        std::vector<uint16_t> w(a.size());
        for (size_t i = 0; i < a.size(); ++i) w[i] = f32_to_f16(a[i] + b[i]);
        if (!dalloc(patch_w_, w.size(), err)) return false;
        VK(cudaMemcpy(patch_w_, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    }
    std::vector<float> ms = {g.mean[0], g.mean[1], g.mean[2]}, sd = {g.std[0], g.std[1], g.std[2]};
    if (!upload_f32(mean_, ms, err) || !upload_f32(std_, sd, err)) return false;
    // the work buffers, for max_tokens blocks
    max_n_ = g.max_tokens * 4;
    const size_t N = (size_t) max_n_;
    rows_ff_ = std::min(max_n_, 2560);
    rows_pf_ = 1024;
    slot_n_ = std::max({per_block, 2 * O * PF, MR * O});
    const size_t gu = std::max((size_t) rows_ff_ * F, (size_t) rows_pf_ * PF);
    if (!dalloc(slot_, slot_n_, err) || !dalloc(x_, N * D, err) || !dalloc(y_, N * D, err) || !dalloc(qkv_, N * 3 * D, err) ||
        !dalloc(xn_, N * D, err) || !dalloc(o_, N * D, err) || !dalloc(gb_, gu, err) || !dalloc(ub_, gu, err) ||
        !dalloc(mid_, (size_t) rows_ff_ * F, err) || !dalloc(pt_, N * pe, err) ||
        !dalloc(rgb_, N * (size_t) g.patch * g.patch * 3, err))
        return false;
    cudaStream_t st = nullptr;
    VK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    st_ = st;
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS || cublasSetStream(h, st) != CUBLAS_STATUS_SUCCESS) {
        err = "glm5 vision: cublasCreate failed";
        return false;
    }
    blas_ = h;
    return true;
}

bool Glm5Vision::gemm(const void* W, const void* x, float* y, int rows, int in, int out, int ldy, std::string& err) {
    const float one = 1.0f, zero = 0.0f;
    const auto e = cublasGemmEx((cublasHandle_t) blas_, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &one, W, CUDA_R_16F, in, x, CUDA_R_16F, in,
                                &zero, y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (e != CUBLAS_STATUS_SUCCESS) {
        err = "glm5 vision: cublasGemmEx: status " + std::to_string((int) e);
        return false;
    }
    return true;
}
bool Glm5Vision::fetch(size_t off, size_t n, void* dst, std::string& err) {
    VK(cudaMemcpyAsync(dst, host_ + off, n * 2, cudaMemcpyHostToDevice, (cudaStream_t) st_));
    return true;
}

bool Glm5Vision::encode(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& rows, std::string& err) {
    const int cell = g_.patch * g_.merge, D = g_.dim, F = g_.ff, O = g_.out, PF = g_.proj_ff;
    if (width <= 0 || height <= 0 || width % cell != 0 || height % cell != 0) {
        err = "glm5 vision: a picture of " + std::to_string(width) + " x " + std::to_string(height);
        return false;
    }
    const int bw = width / cell, bh = height / cell, M = bw * bh, N = 4 * M;
    if (N > max_n_) {
        err = "glm5 vision: " + std::to_string(M) + " image tokens, at most " + std::to_string(g_.max_tokens);
        return false;
    }
    VK(cudaSetDevice(dev_));
    cudaStream_t s = (cudaStream_t) st_;
    const int pe = 3 * g_.patch * g_.patch;
    VK(cudaMemcpyAsync(rgb_, rgb, (size_t) width * height * 3, cudaMemcpyHostToDevice, s));
    patchify_kernel<<<N, 256, 0, s>>>(rgb_, width, bw, g_.patch, mean_, std_, pt_);
    if (!gemm(patch_w_, pt_, x_, N, pe, D, D, err)) return false;
    add_bias_kernel<<<N, 256, 0, s>>>(x_, patch_b_, D);
    const float scale = 0.125f * 1.44269504f;
    for (const Block& k : blocks_) {
        uint16_t* wq = slot_;
        uint16_t* wo = wq + (size_t) 3 * D * D;
        uint16_t* wg = wo + (size_t) D * D;
        uint16_t* wu = wg + (size_t) D * F;
        uint16_t* wd = wu + (size_t) D * F;
        if (!fetch(k.qkv, (size_t) 3 * D * D, wq, err) || !fetch(k.o, (size_t) D * D, wo, err) || !fetch(k.gate, (size_t) D * F, wg, err) ||
            !fetch(k.up, (size_t) D * F, wu, err) || !fetch(k.down, (size_t) D * F, wd, err))
            return false;
        rms_f16_kernel<<<N, 256, 0, s>>>(x_, k.ln1, D, g_.eps, xn_);
        if (!gemm(wq, xn_, qkv_, N, D, 3 * D, 3 * D, err)) return false;
        qk_prep_kernel<<<dim3((unsigned) N, 3), 1024, 0, s>>>(qkv_, k.qkv_b, k.qn, k.kn, bw, g_.theta, g_.eps);
        attn_kernel<<<dim3((unsigned) ((N + 63) / 64), 16), 256, 0, s>>>(qkv_, N, scale, o_);
        if (!gemm(wo, o_, y_, N, D, D, D, err)) return false;
        add_res_kernel<<<N, 256, 0, s>>>(x_, y_, k.o_b, D);
        rms_f16_kernel<<<N, 256, 0, s>>>(x_, k.ln2, D, g_.eps, xn_);
        for (int a = 0; a < N; a += rows_ff_) {
            const int n = std::min(rows_ff_, N - a);
            if (!gemm(wg, xn_ + (size_t) a * D, gb_, n, D, F, F, err) || !gemm(wu, xn_ + (size_t) a * D, ub_, n, D, F, F, err)) return false;
            swiglu_kernel<<<n, 256, 0, s>>>(gb_, ub_, k.gate_b, k.up_b, F, g_.clamp, mid_);
            if (!gemm(wd, mid_, y_ + (size_t) a * D, n, F, D, D, err)) return false;
        }
        add_res_kernel<<<N, 256, 0, s>>>(x_, y_, k.down_b, D);
    }
    rms_f16_kernel<<<N, 256, 0, s>>>(x_, post_ln_, D, g_.eps, xn_);
    // xn_ [N][D] is [M][4 D]: a block's four patches in a row
    const int MR = 4 * D;
    if (!fetch(m_merge_, (size_t) MR * O, slot_, err) || !gemm(slot_, xn_, y_, M, MR, O, O, err)) return false;
    to_f16_kernel<<<M, 256, 0, s>>>(y_, merge_b_, O, o_);
    if (!fetch(m_fc_, (size_t) O * O, slot_, err) || !gemm(slot_, o_, x_, M, O, O, O, err)) return false;
    ln_gelu_kernel<<<M, 256, 0, s>>>(x_, pn_w_, pn_b_, O, 1e-5f, xn_);
    uint16_t* wg = slot_;
    uint16_t* wu = wg + (size_t) O * PF;
    if (!fetch(m_gate_, (size_t) O * PF, wg, err) || !fetch(m_up_, (size_t) O * PF, wu, err)) return false;
    uint16_t* all = (uint16_t*) qkv_;   // ([M][PF] F16: the qkv rows are free now and hold it)
    for (int a = 0; a < M; a += rows_pf_) {
        const int n = std::min(rows_pf_, M - a);
        if (!gemm(wg, xn_ + (size_t) a * O, gb_, n, O, PF, PF, err) || !gemm(wu, xn_ + (size_t) a * O, ub_, n, O, PF, PF, err)) return false;
        swiglu_kernel<<<n, 256, 0, s>>>(gb_, ub_, nullptr, nullptr, PF, g_.clamp, all + (size_t) a * PF);
    }
    if (!fetch(m_down_, (size_t) PF * O, slot_, err) || !gemm(slot_, all, x_, M, PF, O, O, err)) return false;
    to_bf16_kernel<<<M, 256, 0, s>>>(x_, O, o_);
    rows.resize((size_t) M * O);
    VK(cudaMemcpyAsync(rows.data(), o_, rows.size() * 2, cudaMemcpyDeviceToHost, s));
    VK(cudaStreamSynchronize(s));
    return true;
}

}  // namespace strata::ds4
