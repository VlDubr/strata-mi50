// src/dsv41/glm5_kernels.cuh - GLM-5.3-Flash's own device kernels: KDA linear attention, absorbed sparse MLA with the
// k-pool indexer, the sigmoid router.  The math follows llama.cpp's src/models/glm5-next.cpp and ggml-cpu's recurrence.
// A window is T <= kWin tokens; everything here takes T rows, so a decode step and a prompt window are the same calls.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace strata::ds4::glm5 {

// ---- shared
/// Every row of pre (stride 8) to 1 / hc: the mean of the streams through hc_pre_norm, GLM's collapse before the head.
void fill_mean_pre(float* pre, int T, int hc, cudaStream_t s);
/// The window's token rows from a Q8_0 embedding table into the hc streams, in F32: h[t][j][:] = row(tok[t]).
void embed_q8(const uint8_t* table, const int* tok, int T, int dim, int hc, float* h, cudaStream_t s);
/// y[r] = x[r] / rms(x[r]) * w over n values, rows apart by stride.  y may be x.
void rms_rows(const float* x, const float* w, int n, int rows, size_t stride, float eps, float* y, cudaStream_t s);
/// A batched Q8_0 matrix: y[t][h][r] = sum_c W[h][r][c] x[t][h][c] (W [heads][rows][k], x rows x_stride apart).
void bmv_q8(const uint8_t* W, const float* x, size_t x_stride, int heads, int rows, int k, int T, float* y,
            size_t y_stride, cudaStream_t s);

// ---- KDA
/// One of q, k, v through its causal depthwise conv and SiLU, in place, over a window: x [T][d_inner], conv [d_inner][4]
/// (tap 3 the current token), state [3][d_inner] (the last inputs, oldest first).  l2: per-head l2 norm (q and k).
void kda_conv(float* x, const float* conv, float* state, int T, int heads, int hd, bool l2, cudaStream_t s);
/// The recurrence over a window.  q, k, v, graw [T][heads * hd]; beta_raw [T][heads]; a [heads] (= -exp(A_log));
/// dt [heads * hd]; S [heads][hd (value j)][hd (key i)] F32, updated; out [T][heads * hd].
void kda_scan(const float* q, const float* k, const float* v, const float* graw, const float* beta_raw, const float* a,
              const float* dt, float lower_bound, float* S, int T, int heads, int hd, float* out, cudaStream_t s);
/// o[t][h][:] = rms(o[t][h][:]) * norm * sigmoid(g2[t][h][:]), in place.
void kda_out_norm(float* o, const float* g2, const float* norm, int T, int heads, int hd, float eps, cudaStream_t s);

// ---- MLA + indexer
/// The window's latents into the cache: c[p0 + t] = F16(rms(kv[t]) * w).
void mla_latent(const float* kv, const float* w, int n, int T, float eps, uint16_t* c, int64_t p0, cudaStream_t s);
/// The window's key | gate rows into the indexer cache: kg[p0 + t] = F16(LayerNorm(ik[t]) * w + b | ig[t]).
void idx_rows(const float* ik, const float* ig, const float* w, const float* b, int hd, int T, float eps, uint16_t* kg,
              int64_t p0, cudaStream_t s);
/// Pools [first, first + n) of kpool tokens: pooled[p][d] = F16(sum_k softmax_k(gate_k[d] + ape[k][d]) key_k[d]).
void idx_pool(const uint16_t* kg, const float* ape, int hd, int kpool, int64_t first, int n, uint16_t* pooled,
              cudaStream_t s);
/// score[t][p] = sum_h w[t][h] relu(iq[t][h] . pooled[p]) for the pools complete at p0 + t (rows stride apart).  n_max: the widest row.
void idx_score(const float* iq, const float* w, size_t w_stride, const uint16_t* pooled, int heads, int hd, int T,
               int64_t p0, int kpool, int n_max, float w_scale, float* score, long long stride, cudaStream_t s);
/// The sparse attention: for token t at p0 + t, the selected pools' tokens (sel row t, at most top_n pools) and the incomplete
/// tail of its pool.  qa [T][heads][r]; c the latent cache F16 [..][r]; out [T][heads][r].
void mla_attn(const float* qa, const uint16_t* c, const int* sel, int sel_stride, int64_t p0, int top_n,
              int kpool, int heads, int r, int T, float scale, float* out, cudaStream_t s);

// ---- the router
/// Sigmoid scores, the top k of score + bias picked, the unbiased scores normalised and scaled.  logits rows are ld apart.
void route(const float* logits, int ld, const float* bias, int n, int k, int T, float scale, int* idx, float* w,
           int picks_stride, cudaStream_t s);

}  // namespace strata::ds4::glm5
