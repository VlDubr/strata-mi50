// src/dsv41/glm5_kernels.cu - see glm5_kernels.cuh.  A "warp" here is 32 lanes (the engine's logical warp, on wave64 too).
#include "glm5_kernels.cuh"

#include <cuda_fp16.h>

#include <cfloat>
#include <cmath>

namespace strata::ds4::glm5 {
namespace {

constexpr int kMaxT = 16;        // kWin
constexpr int kQ8 = 34;          // a Q8_0 block's bytes

__device__ __forceinline__ float h2f(uint16_t v) { return __half2float(__ushort_as_half(v)); }
__device__ __forceinline__ uint16_t f2h(float v) { return __half_as_ushort(__float2half(v)); }
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// the block's sum (a multiple of 32 threads, at most 1024); every thread gets it
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) part[wid] = v;
    __syncthreads();
    float r = 0.0f;
    for (int i = 0; i < nw; ++i) r += part[i];
    return r;
}
__device__ __forceinline__ float block_max(float v) {
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if (lane == 0) part[wid] = v;
    __syncthreads();
    float r = -FLT_MAX;
    for (int i = 0; i < nw; ++i) r = fmaxf(r, part[i]);
    return r;
}
// a Q8_0 block's scale; the block is 2-byte aligned at best
__device__ __forceinline__ float q8_scale(const uint8_t* b) { return h2f((uint16_t) (b[0] | b[1] << 8)); }

__global__ void fill_kernel(float* pre, int T, float v, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * 8) pre[i] = (i & 7) < hc ? v : 0.0f;
}

__global__ void embed_over_kernel(const uint16_t* __restrict__ rows, const int* __restrict__ at, int dim, int hc, float* h) {
    const uint16_t* r = rows + (size_t) blockIdx.x * dim;
    float* o = h + (size_t) at[blockIdx.x] * hc * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = __uint_as_float((uint32_t) r[i] << 16);
        for (int j = 0; j < hc; ++j) o[(size_t) j * dim + i] = v;
    }
}

__global__ void embed_q8_kernel(const uint8_t* __restrict__ table, const int* __restrict__ tok, int dim, int hc,
                                float* __restrict__ h) {
    const int t = blockIdx.x;
    const size_t rb = (size_t) dim / 32 * kQ8;
    const uint8_t* row = table + (size_t) tok[t] * rb;
    float* out = h + (size_t) t * hc * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const uint8_t* b = row + (size_t) (i / 32) * kQ8;
        const float v = q8_scale(b) * (float) (int8_t) b[2 + i % 32];
        for (int j = 0; j < hc; ++j) out[(size_t) j * dim + i] = v;
    }
}

__global__ void rms_kernel(const float* x, const float* __restrict__ w, int n, size_t stride, float eps, float* y) {
    const float* r = x + (size_t) blockIdx.x * stride;
    float* o = y + (size_t) blockIdx.x * stride;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += r[i] * r[i];
    const float inv = rsqrtf(block_sum(ss) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = r[i] * inv * w[i];
}

// grid (rows / 8, heads), 256 threads: a warp a row, each lane 8 values of a 32-value block in turn; the head's window of
// activations staged in shared memory
__global__ void __launch_bounds__(256) bmv_q8_kernel(const uint8_t* __restrict__ W, const float* __restrict__ x,
                                                     size_t x_stride, int rows, int k, int T, float* __restrict__ y,
                                                     size_t y_stride) {
    extern __shared__ float xs[];   // [T][k]
    const int h = blockIdx.y;
    for (int i = threadIdx.x; i < T * k; i += blockDim.x)
        xs[i] = x[(size_t) (i / k) * x_stride + (size_t) h * k + i % k];
    __syncthreads();
    const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const size_t rb = (size_t) k / 32 * kQ8;
    const uint8_t* w = W + ((size_t) h * rows + row) * rb;
    float acc[kMaxT];
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) acc[t] = 0.0f;
    const int nb = k / 32, sub = lane & 3;
    for (int b = lane >> 2; b < nb; b += 8) {
        const uint8_t* blk = w + (size_t) b * kQ8;
        const float d = q8_scale(blk);
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = d * (float) (int8_t) blk[2 + sub * 8 + i];
        const int c0 = b * 32 + sub * 8;
#pragma unroll
        for (int t = 0; t < kMaxT; ++t)
            if (t < T) {
                float a = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) a = fmaf(v[i], xs[t * k + c0 + i], a);
                acc[t] += a;
            }
    }
#pragma unroll
    for (int t = 0; t < kMaxT; ++t)
        if (t < T) {
            const float r = warp_sum(acc[t]);
            if (lane == 0) y[(size_t) t * y_stride + (size_t) h * rows + row] = r;
        }
}

// grid heads, hd threads: thread c is channel c of its head, the window in order
struct Conv3 {
    float* x[3];
    const float* conv[3];
};
__global__ void kda_conv_kernel(Conv3 a, float* state, int T, int d_inner, float* snap, size_t snap_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, which = blockIdx.y, l2 = which < 2;   // (q, k, v: q and k normed)
    float* x = a.x[which];
    const float* __restrict__ conv = a.conv[which];
    state += (size_t) which * 3 * d_inner;
    if (snap != nullptr) snap += (size_t) which * 3 * d_inner;
    const float w0 = conv[(size_t) c * 4], w1 = conv[(size_t) c * 4 + 1], w2 = conv[(size_t) c * 4 + 2],
                w3 = conv[(size_t) c * 4 + 3];
    float s0 = state[c], s1 = state[(size_t) d_inner + c], s2 = state[(size_t) 2 * d_inner + c];
    for (int t = 0; t < T; ++t) {
        const float in = x[(size_t) t * d_inner + c];
        const float a = w0 * s0 + w1 * s1 + w2 * s2 + w3 * in;
        float v = a * sigmoidf(a);
        if (l2) v *= rsqrtf(block_sum(v * v) + 1e-6f);
        x[(size_t) t * d_inner + c] = v;
        s0 = s1;
        s1 = s2;
        s2 = in;
        if (snap != nullptr && t + 1 < T) {
            float* o = snap + (size_t) t * snap_stride;
            o[c] = s0;
            o[(size_t) d_inner + c] = s1;
            o[(size_t) 2 * d_inner + c] = s2;
        }
    }
    state[c] = s0;
    state[(size_t) d_inner + c] = s1;
    state[(size_t) 2 * d_inner + c] = s2;
}

// grid (heads, hd / 64), 64 threads: thread j owns column j of S (S[:, j], all its key rows, in registers), so nothing but
// the token's k, q and decay is shared.  ggml-cpu's order: decay, delta, outer product, output
template <int HD>
__global__ void __launch_bounds__(64) kda_scan_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                                      const float* __restrict__ v, const float* __restrict__ graw,
                                                      const float* __restrict__ beta_raw, const float* __restrict__ a,
                                                      const float* __restrict__ dt, float lb, float* S, int T, int heads,
                                                      float* __restrict__ out, float* __restrict__ snap) {
    __shared__ float sk[HD], sq[HD], sg[HD];
    const int h = blockIdx.x, j = blockIdx.y * 64 + threadIdx.x;
    const size_t di = (size_t) heads * HD;
    float* col = S + (size_t) h * HD * HD + j;   // ([key i][value j]: a wave's lanes read one row)
    float st[HD];
#pragma unroll
    for (int i = 0; i < HD; ++i) st[i] = col[(size_t) i * HD];
    const float na = -a[h];   // exp(A_log)
    const float scale = rsqrtf((float) HD);
    for (int t = 0; t < T; ++t) {
        const size_t base = (size_t) t * di + (size_t) h * HD;
        __syncthreads();
        for (int i = threadIdx.x; i < HD; i += 64) {
            sk[i] = k[base + i];
            sq[i] = q[base + i];
            const float g = graw[base + i] + dt[(size_t) h * HD + i];
            sg[i] = expf(lb * sigmoidf(na * g));
        }
        __syncthreads();
        const float beta = sigmoidf(beta_raw[(size_t) t * heads + h]);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < HD; ++i) {
            st[i] *= sg[i];
            sum = fmaf(st[i], sk[i], sum);
        }
        const float delta = (v[base + j] - sum) * beta;
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < HD; ++i) {
            st[i] = fmaf(sk[i], delta, st[i]);
            o = fmaf(st[i], sq[i], o);
        }
        out[base + j] = o * scale;
        if (snap != nullptr && t + 1 < T) {   // (a verify window: what a rejected row goes back to)
            float* sc = snap + (size_t) t * heads * HD * HD + (size_t) h * HD * HD + j;
#pragma unroll
            for (int i = 0; i < HD; ++i) sc[(size_t) i * HD] = st[i];
        }
    }
#pragma unroll
    for (int i = 0; i < HD; ++i) col[(size_t) i * HD] = st[i];
}

// grid T, 256 threads
__global__ void mtp_in_kernel(const float* __restrict__ e, const float* __restrict__ h, int ring, int first, int zero_first,
                              const float* __restrict__ enorm, const float* __restrict__ hnorm, int dim, float eps,
                              float* __restrict__ x) {
    const int t = blockIdx.x;
    const float* er = e + (size_t) t * dim;
    const float* hr = h + (size_t) ((first + t) % ring) * dim;
    const bool none = zero_first != 0 && t == 0;
    float se = 0.0f, sh = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        se += er[i] * er[i];
        if (!none) sh += hr[i] * hr[i];
    }
    se = block_sum(se);
    sh = block_sum(sh);
    const float ie = rsqrtf(se / dim + eps), ih = rsqrtf(sh / dim + eps);
    float* o = x + (size_t) t * 2 * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        o[i] = er[i] * ie * enorm[i];
        o[dim + i] = none ? 0.0f : hr[i] * ih * hnorm[i];
    }
}

// one block of 1024
__global__ void argmax_conf_kernel(const float* __restrict__ l, int n, int* tok, float* conf) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    float best = -FLT_MAX;
    int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (l[i] > best) {
            best = l[i];
            bi = i;
        }
    sv[threadIdx.x] = best;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int o = 512; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) {
            const float v = sv[threadIdx.x + o];
            const int i = si[threadIdx.x + o];
            if (v > sv[threadIdx.x] || (v == sv[threadIdx.x] && i < si[threadIdx.x])) {
                sv[threadIdx.x] = v;
                si[threadIdx.x] = i;
            }
        }
        __syncthreads();
    }
    const float m = sv[0];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += expf(l[i] - m);
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        *tok = si[0];
        *conf = 1.0f / sum;
    }
}

// grid (heads, T), hd threads
__global__ void kda_out_kernel(float* o, const float* __restrict__ g2, const float* __restrict__ norm, int heads, int hd,
                               float eps) {
    const size_t base = ((size_t) blockIdx.y * heads + blockIdx.x) * hd;
    const int i = threadIdx.x;
    const float v = o[base + i];
    const float inv = rsqrtf(block_sum(v * v) / (float) hd + eps);
    o[base + i] = v * inv * norm[i] * sigmoidf(g2[base + i]);
}

__global__ void mla_latent_kernel(const float* __restrict__ kv, const float* __restrict__ w, int n, float eps, uint16_t* c,
                                  int64_t p0) {
    const float* r = kv + (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += r[i] * r[i];
    const float inv = rsqrtf(block_sum(ss) / (float) n + eps);
    uint16_t* o = c + (size_t) (p0 + blockIdx.x) * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = f2h(r[i] * inv * w[i]);
}

// grid T, hd threads
__global__ void idx_rows_kernel(const float* __restrict__ ik, const float* __restrict__ ig, const float* __restrict__ w,
                                const float* __restrict__ b, int hd, float eps, uint16_t* kg, int64_t p0) {
    const int t = blockIdx.x, i = threadIdx.x;
    const float v = ik[(size_t) t * hd + i];
    const float mean = block_sum(v) / (float) hd;
    const float c = v - mean;
    const float inv = rsqrtf(block_sum(c * c) / (float) hd + eps);
    uint16_t* o = kg + (size_t) ((p0 + t) & (kKgRing - 1)) * 2 * hd;
    o[i] = f2h(c * inv * w[i] + b[i]);
    o[hd + i] = f2h(ig[(size_t) t * hd + i]);
}

// grid n, hd threads
__global__ void idx_pool_kernel(const uint16_t* __restrict__ kg, const float* __restrict__ ape, int hd, int kpool,
                                int64_t first, uint16_t* pooled) {
    const int64_t p = first + blockIdx.x;
    const int d = threadIdx.x;
    const auto row = [&](int k) { return kg + (size_t) ((p * kpool + k) & (kKgRing - 1)) * 2 * hd; };
    float m = -FLT_MAX;
    for (int k = 0; k < kpool; ++k) m = fmaxf(m, h2f(row(k)[hd + d]) + ape[(size_t) k * hd + d]);
    float den = 0.0f, num = 0.0f;
    for (int k = 0; k < kpool; ++k) {
        const float e = expf(h2f(row(k)[hd + d]) + ape[(size_t) k * hd + d] - m);
        den += e;
        num += e * h2f(row(k)[d]);
    }
    pooled[(size_t) p * hd + d] = f2h(num / den);
}

// grid (cdiv(n_max, 8), T), 256 threads: a warp a pool, lane l holds dims 4l .. 4l + 3 of it (hd 128)
__global__ void __launch_bounds__(256) idx_score_kernel(const float* __restrict__ iq, const float* __restrict__ w,
                                                        size_t w_stride, const uint16_t* __restrict__ pooled, int heads,
                                                        int hd, int64_t p0, int kpool, float w_scale, float* score,
                                                        long long stride) {
    extern __shared__ float sq[];   // [heads][hd], then [heads] weights
    const int t = blockIdx.y;
    const int n = (int) ((p0 + t + 1) / kpool);
    if ((int) (blockIdx.x * 8) >= n) return;
    for (int i = threadIdx.x; i < heads * hd; i += blockDim.x) sq[i] = iq[(size_t) t * heads * hd + i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) sq[heads * hd + i] = w[(size_t) t * w_stride + i] * w_scale;
    __syncthreads();
    const int p = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (p >= n) return;
    const uint16_t* r = pooled + (size_t) p * hd + lane * 4;
    const float k0 = h2f(r[0]), k1 = h2f(r[1]), k2 = h2f(r[2]), k3 = h2f(r[3]);
    float s = 0.0f;
    for (int h = 0; h < heads; ++h) {
        const float* qh = sq + h * hd + lane * 4;
        const float dot = warp_sum(k0 * qh[0] + k1 * qh[1] + k2 * qh[2] + k3 * qh[3]);
        s += sq[heads * hd + h] * fmaxf(dot, 0.0f);
    }
    if (lane == 0) score[(long long) t * stride + p] = s;
}

// grid (heads, T), 256 threads.  sc: a score a selected token (dynamic shared memory).  The token list is implicit: the
// selected pools' members, then the tail.
template <int R>
__global__ void __launch_bounds__(256) mla_attn_kernel(const float* __restrict__ qa, const uint16_t* __restrict__ c,
                                                       const int* __restrict__ sel, int sel_stride, int64_t p0, int top_n,
                                                       int kpool, int heads, float scale,
                                                       float* __restrict__ out) {
    extern __shared__ float sc[];
    __shared__ float sqv[R];
    const int h = blockIdx.x, t = blockIdx.y;
    const float* q = qa + ((size_t) t * heads + h) * R;
    for (int i = threadIdx.x; i < R; i += blockDim.x) sqv[i] = q[i];
    __syncthreads();
    const int* st = sel + (size_t) t * sel_stride;
    const int n_vis = (int) ((p0 + t + 1) / kpool), np = min(n_vis, top_n), n_pool_tok = np * kpool;
    const int a0 = n_vis * kpool, a1 = (int) (p0 + t);
    const int n = n_pool_tok + (a1 >= a0 ? a1 - a0 + 1 : 0);
    auto token = [&](int i) { return i < n_pool_tok ? st[i / kpool] * kpool + i % kpool : a0 + (i - n_pool_tok); };
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int i = wid; i < n; i += 8) {
        const uint16_t* row = c + (size_t) token(i) * R;
        float a = 0.0f;
        for (int d = lane; d < R; d += 32) a = fmaf(h2f(row[d]), sqv[d], a);
        a = warp_sum(a);
        if (lane == 0) sc[i] = a * scale;
    }
    __syncthreads();
    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, sc[i]);
    m = block_max(m);
    float den = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = expf(sc[i] - m);
        sc[i] = e;
        den += e;
    }
    den = block_sum(den);   // (its barriers also order the sc writes before the reads below)
    constexpr int PER = R / 256;
    float acc[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) acc[j] = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float p = sc[i];
        const uint16_t* row = c + (size_t) token(i) * R;
#pragma unroll
        for (int j = 0; j < PER; ++j) acc[j] = fmaf(p, h2f(row[threadIdx.x + j * 256]), acc[j]);
    }
    float* o = out + ((size_t) t * heads + h) * R;
#pragma unroll
    for (int j = 0; j < PER; ++j) o[threadIdx.x + j * 256] = n > 0 ? acc[j] / den : 0.0f;
}

__global__ void rows_add_kernel(const float* __restrict__ rows, RowsAdd a, int n, int dim, float* out) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= dim) return;
    for (int r = 0; r < n; ++r) out[(size_t) a.tok[r] * dim + e] += a.w[r] * rows[(size_t) r * dim + e];
}

__global__ void add_kernel(const float* __restrict__ x, float* y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

// grid T, 512 threads (n <= 512)
__global__ void __launch_bounds__(512) route_kernel(const float* __restrict__ logits, int ld,
                                                    const float* __restrict__ bias, int n, int k, float scale,
                                                    int* __restrict__ idx, float* __restrict__ w, int stride) {
    __shared__ float bv[16];
    __shared__ int bi[16];
    __shared__ float sel_p[32];
    __shared__ int sel_i[32];
    const int t = blockIdx.x, i = threadIdx.x;
    const float p = i < n ? sigmoidf(logits[(size_t) t * ld + i]) : 0.0f;
    float key = i < n ? p + bias[i] : -FLT_MAX;
    for (int r = 0; r < k; ++r) {
        float v = key;
        int ix = i;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
            if (ov > v || (ov == v && oi < ix)) {
                v = ov;
                ix = oi;
            }
        }
        if ((i & 31) == 0) {
            bv[i >> 5] = v;
            bi[i >> 5] = ix;
        }
        __syncthreads();
        float best = bv[0];
        int bix = bi[0];
        for (int q = 1; q < (int) (blockDim.x >> 5); ++q)
            if (bv[q] > best || (bv[q] == best && bi[q] < bix)) {
                best = bv[q];
                bix = bi[q];
            }
        if (i == bix) {
            key = -FLT_MAX;
            sel_p[r] = p;
            sel_i[r] = i;
        }
        __syncthreads();
    }
    if (i == 0) {
        float sum = 0.0f;
        for (int r = 0; r < k; ++r) sum += sel_p[r];
        sum = fmaxf(sum, 6.103515625e-5f);
        for (int r = 0; r < k; ++r) {
            idx[(size_t) t * stride + r] = sel_i[r];
            w[(size_t) t * stride + r] = sel_p[r] / sum * scale;
        }
    }
}

}  // namespace

void fill_mean_pre(float* pre, int T, int hc, cudaStream_t s) {
    fill_kernel<<<(T * 8 + 127) / 128, 128, 0, s>>>(pre, T, 1.0f / (float) hc, hc);
}
void embed_q8(const uint8_t* table, const int* tok, int T, int dim, int hc, float* h, cudaStream_t s) {
    embed_q8_kernel<<<T, 256, 0, s>>>(table, tok, dim, hc, h);
}
void embed_over(const uint16_t* rows, const int* at, int n, int dim, int hc, float* h, cudaStream_t s) {
    if (n > 0) embed_over_kernel<<<n, 256, 0, s>>>(rows, at, dim, hc, h);
}
void rms_rows(const float* x, const float* w, int n, int rows, size_t stride, float eps, float* y, cudaStream_t s) {
    rms_kernel<<<rows, 256, 0, s>>>(x, w, n, stride, eps, y);
}
void bmv_q8(const uint8_t* W, const float* x, size_t x_stride, int heads, int rows, int k, int T, float* y,
            size_t y_stride, cudaStream_t s) {
    bmv_q8_kernel<<<dim3((unsigned) ((rows + 7) / 8), (unsigned) heads), 256, (size_t) T * k * 4, s>>>(W, x, x_stride, rows, k,
                                                                                                       T, y, y_stride);
}
void kda_conv(float* xq, float* xk, float* xv, const float* cq, const float* ck, const float* cv, float* state, int T, int heads, int hd,
              cudaStream_t s, float* snap, size_t snap_stride) {
    Conv3 a{{xq, xk, xv}, {cq, ck, cv}};
    kda_conv_kernel<<<dim3((unsigned) heads, 3), hd, 0, s>>>(a, state, T, heads * hd, snap, snap_stride);
}
void mtp_in(const float* e, const float* h, int ring, int first, bool zero_first, const float* enorm, const float* hnorm,
            int dim, int T, float eps, float* x, cudaStream_t s) {
    mtp_in_kernel<<<T, 256, 0, s>>>(e, h, ring, first, zero_first ? 1 : 0, enorm, hnorm, dim, eps, x);
}
void argmax_conf(const float* logits, int n, int* tok, float* conf, cudaStream_t s) {
    argmax_conf_kernel<<<1, 1024, 0, s>>>(logits, n, tok, conf);
}
void kda_scan(const float* q, const float* k, const float* v, const float* graw, const float* beta_raw, const float* a,
              const float* dt, float lower_bound, float* S, int T, int heads, int hd, float* out, cudaStream_t s, float* snap) {
    (void) hd;   // (128, the only head size: the state column lives in registers)
    kda_scan_kernel<128><<<dim3((unsigned) heads, 2), 64, 0, s>>>(q, k, v, graw, beta_raw, a, dt, lower_bound, S, T, heads, out, snap);
}
void kda_out_norm(float* o, const float* g2, const float* norm, int T, int heads, int hd, float eps, cudaStream_t s) {
    kda_out_kernel<<<dim3((unsigned) heads, (unsigned) T), hd, 0, s>>>(o, g2, norm, heads, hd, eps);
}
void mla_latent(const float* kv, const float* w, int n, int T, float eps, uint16_t* c, int64_t p0, cudaStream_t s) {
    mla_latent_kernel<<<T, 256, 0, s>>>(kv, w, n, eps, c, p0);
}
void idx_rows(const float* ik, const float* ig, const float* w, const float* b, int hd, int T, float eps, uint16_t* kg,
              int64_t p0, cudaStream_t s) {
    idx_rows_kernel<<<T, hd, 0, s>>>(ik, ig, w, b, hd, eps, kg, p0);
}
void idx_pool(const uint16_t* kg, const float* ape, int hd, int kpool, int64_t first, int n, uint16_t* pooled,
              cudaStream_t s) {
    if (n > 0) idx_pool_kernel<<<n, hd, 0, s>>>(kg, ape, hd, kpool, first, pooled);
}
void idx_score(const float* iq, const float* w, size_t w_stride, const uint16_t* pooled, int heads, int hd, int T,
               int64_t p0, int kpool, int n_max, float w_scale, float* score, long long stride, cudaStream_t s) {
    if (n_max <= 0) return;
    idx_score_kernel<<<dim3((unsigned) ((n_max + 7) / 8), (unsigned) T), 256, (size_t) (heads * hd + heads) * 4, s>>>(
        iq, w, w_stride, pooled, heads, hd, p0, kpool, w_scale, score, stride);
}
void mla_attn(const float* qa, const uint16_t* c, const int* sel, int sel_stride, int64_t p0, int top_n,
              int kpool, int heads, int r, int T, float scale, float* out, cudaStream_t s) {
    (void) r;   // (512, the only latent width)
    const size_t sh = (size_t) (2048 + 64) * 4;
    mla_attn_kernel<512><<<dim3((unsigned) heads, (unsigned) T), 256, sh, s>>>(qa, c, sel, sel_stride, p0, top_n, kpool,
                                                                               heads, scale, out);
}
void rows_add(const float* rows, const RowsAdd& a, int n, int dim, float* out, cudaStream_t s) {
    rows_add_kernel<<<(dim + 255) / 256, 256, 0, s>>>(rows, a, n, dim, out);
}
void add(const float* x, float* y, size_t n, cudaStream_t s) {
    add_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, y, n);
}
void route(const float* logits, int ld, const float* bias, int n, int k, int T, float scale, int* idx, float* w,
           int picks_stride, cudaStream_t s) {
    route_kernel<<<T, 512, 0, s>>>(logits, ld, bias, n, k, scale, idx, w, picks_stride);
}

}  // namespace strata::ds4::glm5
