// tests/dsv41/glm5_kernels_test.cu - GLM-5.3-Flash's device kernels against plain double-precision references: the KDA
// conv and recurrence across two windows (the state carried over), the batched Q8_0 product, the pool, the sparse attention
// with its tail, and the router.  Needs a GPU, no model.
#include "../../src/dsv41/glm5_kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::ds4;

static int failed = 0;
static std::mt19937 rng(7);
static float rnd(float a = 1.0f) { return std::uniform_real_distribution<float>(-a, a)(rng); }

template <class T> static T* up(const std::vector<T>& v) {
    T* p = nullptr;
    cudaMalloc(&p, v.size() * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> static std::vector<T> down(const T* p, size_t n) {
    std::vector<T> v(n);
    cudaDeviceSynchronize();
    cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}
static void check(const char* what, const std::vector<float>& got, const std::vector<double>& want, double tol) {
    double worst = 0;
    for (size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::fabs((double) got[i] - want[i]));
    const bool ok = worst <= tol && std::isfinite(worst);
    std::printf("%-28s worst %.3e (tol %.1e) %s\n", what, worst, tol, ok ? "ok" : "FAILED");
    failed += !ok;
}
static uint16_t f16(float v) { const __half h = __float2half(v); uint16_t u; std::memcpy(&u, &h, 2); return u; }
static float f32(uint16_t u) { __half h; std::memcpy(&h, &u, 2); return __half2float(h); }
static double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }

static void test_kda() {
    const int H = 3, HD = 128, DI = H * HD;
    const int T1 = 5, T2 = 3, T = T1 + T2;
    std::vector<float> raw[3], conv[3], graw((size_t) T * DI), beta((size_t) T * H), a(H), dt(DI);
    for (int c = 0; c < 3; ++c) {
        raw[c].resize((size_t) T * DI);
        conv[c].resize((size_t) DI * 4);
        for (auto& v : raw[c]) v = rnd();
        for (auto& v : conv[c]) v = rnd(0.5f);
    }
    for (auto& v : graw) v = rnd();
    for (auto& v : beta) v = rnd(2.0f);
    for (auto& v : a) v = -std::exp(rnd(0.5f));
    for (auto& v : dt) v = rnd(0.5f);
    const float lb = -5.0f;
    // the reference
    std::vector<double> qkv[3];
    for (int c = 0; c < 3; ++c) {
        qkv[c].assign((size_t) T * DI, 0.0);
        for (int i = 0; i < DI; ++i)
            for (int t = 0; t < T; ++t) {
                double y = 0;
                for (int k = 0; k < 4; ++k) {
                    const int u = t - 3 + k;
                    y += u >= 0 ? (double) conv[c][(size_t) i * 4 + k] * raw[c][(size_t) u * DI + i] : 0.0;
                }
                qkv[c][(size_t) t * DI + i] = y * sig(y);
            }
        if (c < 2)
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < H; ++h) {
                    double ss = 0;
                    for (int i = 0; i < HD; ++i) ss += qkv[c][(size_t) t * DI + h * HD + i] * qkv[c][(size_t) t * DI + h * HD + i];
                    const double inv = 1.0 / std::sqrt(ss + 1e-6);
                    for (int i = 0; i < HD; ++i) qkv[c][(size_t) t * DI + h * HD + i] *= inv;
                }
    }
    std::vector<double> S((size_t) H * HD * HD, 0.0), want((size_t) T * DI);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            double* s = S.data() + (size_t) h * HD * HD;
            const size_t b = (size_t) t * DI + h * HD;
            const double be = sig(beta[(size_t) t * H + h]);
            for (int i = 0; i < HD; ++i) {
                const double e = std::exp(lb * sig(-(double) a[h] * ((double) graw[b + i] + dt[h * HD + i])));
                for (int j = 0; j < HD; ++j) s[i * HD + j] *= e;
            }
            for (int j = 0; j < HD; ++j) {
                double sum = 0;
                for (int i = 0; i < HD; ++i) sum += s[i * HD + j] * qkv[1][b + i];
                const double dl = (qkv[2][b + j] - sum) * be;
                for (int i = 0; i < HD; ++i) s[i * HD + j] += qkv[1][b + i] * dl;
            }
            for (int j = 0; j < HD; ++j) {
                double o = 0;
                for (int i = 0; i < HD; ++i) o += s[i * HD + j] * qkv[0][b + i];
                want[b + j] = o / std::sqrt((double) HD);
            }
        }
    // the device, in two windows
    float *dq = up(raw[0]), *dk = up(raw[1]), *dv = up(raw[2]), *dc0 = up(conv[0]), *dc1 = up(conv[1]), *dc2 = up(conv[2]);
    float *dg = up(graw), *db = up(beta), *da = up(a), *ddt = up(dt);
    std::vector<float> zs((size_t) H * HD * HD, 0.0f), zc((size_t) 9 * DI, 0.0f), zo((size_t) T * DI, 0.0f);
    float *dS = up(zs), *dst = up(zc), *dout = up(zo);
    int t0 = 0;
    for (int Tw : {T1, T2}) {
        const size_t o = (size_t) t0 * DI;
        glm5::kda_conv(dq + o, dc0, dst, Tw, H, HD, true, 0);
        glm5::kda_conv(dk + o, dc1, dst + (size_t) 3 * DI, Tw, H, HD, true, 0);
        glm5::kda_conv(dv + o, dc2, dst + (size_t) 6 * DI, Tw, H, HD, false, 0);
        glm5::kda_scan(dq + o, dk + o, dv + o, dg + o, db + (size_t) t0 * H, da, ddt, lb, dS, Tw, H, HD, dout + o, 0);
        t0 += Tw;
    }
    check("kda conv q", down(dq, (size_t) T * DI), qkv[0], 2e-5);
    check("kda scan", down(dout, (size_t) T * DI), want, 5e-5);
    // a verify window's snapshots: the state after each row but the last is the state a window that ended there leaves
    {
        const size_t ns = (size_t) H * HD * HD, nc = (size_t) 3 * DI;
        std::vector<float> st0(ns, 0.0f), cv0(nc, 0.0f), x((size_t) 2 * DI), g2((size_t) 2 * DI), b2((size_t) 2 * H);
        for (size_t i = 0; i < x.size(); ++i) x[i] = g2[i] = 0.01f * (float) ((i * 37) % 101) - 0.5f;
        for (size_t i = 0; i < b2.size(); ++i) b2[i] = 0.1f * (float) i - 0.3f;
        auto run = [&](int rows, float* snap, float* csnap, std::vector<float>& st_out, std::vector<float>& cv_out) {
            float *q = up(x), *k = up(x), *v = up(x), *g = up(g2), *b = up(b2), *S = up(st0), *C = up(cv0), *o = up(x);
            glm5::kda_conv(q, dc0, C, rows, H, HD, false, 0, csnap, nc);
            glm5::kda_scan(q, k, v, g, b, da, ddt, lb, S, rows, H, HD, o, 0, snap);
            cudaDeviceSynchronize();
            st_out = down(S, ns);
            cv_out = down(C, nc);
        };
        std::vector<float> one_s, one_c, two_s, two_c;
        float *snap = up(st0), *csnap = up(cv0);
        run(2, snap, csnap, two_s, two_c);
        run(1, nullptr, nullptr, one_s, one_c);
        std::vector<double> ws(one_s.begin(), one_s.end()), wc(one_c.begin(), one_c.end());
        check("kda snapshot", down(snap, ns), ws, 0.0);
        check("kda conv snapshot", down(csnap, nc), wc, 0.0);
    }
}

static void test_bmv() {
    const int heads = 3, rows = 40, k = 96, T = 5;
    std::vector<uint8_t> W((size_t) heads * rows * (k / 32) * 34);
    std::vector<double> dq((size_t) heads * rows * k);
    for (size_t b = 0; b < W.size() / 34; ++b) {
        const uint16_t d = f16(0.01f + 0.02f * (float) (b % 7));
        std::memcpy(&W[b * 34], &d, 2);
        for (int i = 0; i < 32; ++i) {
            const int8_t q = (int8_t) (rng() % 255 - 127);
            W[b * 34 + 2 + i] = (uint8_t) q;
            dq[b * 32 + i] = (double) f32(d) * q;
        }
    }
    const size_t xs = (size_t) heads * k + 16;
    std::vector<float> x((size_t) T * xs);
    for (auto& v : x) v = rnd();
    const size_t ys = (size_t) heads * rows;
    std::vector<double> want((size_t) T * ys);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < heads; ++h)
            for (int r = 0; r < rows; ++r) {
                double a = 0;
                for (int c = 0; c < k; ++c) a += dq[((size_t) h * rows + r) * k + c] * x[(size_t) t * xs + (size_t) h * k + c];
                want[(size_t) t * ys + (size_t) h * rows + r] = a;
            }
    uint8_t* dW = up(W);
    float* dx = up(x);
    float* dy = up(std::vector<float>((size_t) T * ys, 0.0f));
    glm5::bmv_q8(dW, dx, xs, heads, rows, k, T, dy, ys, 0);
    check("bmv q8", down(dy, (size_t) T * ys), want, 1e-3);
}

static void test_pool_attn() {
    const int hd = 128, kp = 4, np = 6;
    std::vector<uint16_t> kg((size_t) np * kp * 2 * hd);
    for (auto& v : kg) v = f16(rnd());
    std::vector<float> ape((size_t) kp * hd);
    for (auto& v : ape) v = rnd();
    std::vector<double> want((size_t) np * hd);
    for (int p = 0; p < np; ++p)
        for (int d = 0; d < hd; ++d) {
            double den = 0, num = 0;
            for (int k = 0; k < kp; ++k) {
                const uint16_t* r = kg.data() + (size_t) (p * kp + k) * 2 * hd;
                const double e = std::exp((double) f32(r[hd + d]) + ape[(size_t) k * hd + d]);
                den += e;
                num += e * f32(r[d]);
            }
            want[(size_t) p * hd + d] = num / den;
        }
    uint16_t* dkg = up(kg);
    float* dape = up(ape);
    uint16_t* dp = up(std::vector<uint16_t>((size_t) np * hd, 0));
    glm5::idx_pool(dkg, dape, hd, kp, 0, np, dp, 0);
    const auto got16 = down(dp, (size_t) np * hd);
    std::vector<float> got(got16.size());
    for (size_t i = 0; i < got.size(); ++i) got[i] = f32(got16[i]);
    check("idx pool", got, want, 4e-3);

    // the attention: 23 tokens, query at 22 sees pools {1, 4} (top 2) and the tail 20..22; query at 23 ... its own row
    const int heads = 2, R = 512, n = 24, T = 2, stride = 8, top = 2;
    std::vector<uint16_t> c((size_t) n * R);
    for (auto& v : c) v = f16(rnd());
    std::vector<float> qa((size_t) T * heads * R);
    for (auto& v : qa) v = rnd();
    std::vector<int> sel((size_t) T * stride, 0);
    sel[0] = 1, sel[1] = 4, sel[stride] = 0, sel[stride + 1] = 5;
    const int64_t p0 = 22;
    std::vector<double> wa((size_t) T * heads * R);
    for (int t = 0; t < T; ++t) {
        std::vector<int> tok;
        for (int i = 0; i < top; ++i)
            for (int k = 0; k < kp; ++k) tok.push_back(sel[(size_t) t * stride + i] * kp + k);
        const int nvis = (int) ((p0 + t + 1) / kp);
        for (int u = nvis * kp; u <= p0 + t; ++u) tok.push_back(u);
        for (int h = 0; h < heads; ++h) {
            std::vector<double> sc;
            double m = -1e30;
            for (int u : tok) {
                double a = 0;
                for (int d = 0; d < R; ++d) a += (double) f32(c[(size_t) u * R + d]) * qa[((size_t) t * heads + h) * R + d];
                sc.push_back(a / 16.0);
                m = std::max(m, sc.back());
            }
            double den = 0;
            for (double& v : sc) den += (v = std::exp(v - m));
            for (int d = 0; d < R; ++d) {
                double o = 0;
                for (size_t i = 0; i < tok.size(); ++i) o += sc[i] * f32(c[(size_t) tok[i] * R + d]);
                wa[((size_t) t * heads + h) * R + d] = o / den;
            }
        }
    }
    uint16_t* dc = up(c);
    float* dqa = up(qa);
    int* dsel = up(sel);
    float* dout = up(std::vector<float>((size_t) T * heads * R, 0.0f));
    glm5::mla_attn(dqa, dc, dsel, stride, p0, top, kp, heads, R, T, 1.0f / 16.0f, dout, 0);
    check("mla attn", down(dout, (size_t) T * heads * R), wa, 2e-4);
}

static void test_route() {
    const int n = 288, k = 8, T = 4, ld = 1024, picks = 8;
    std::vector<float> lg((size_t) T * ld, 0.0f), bias(n);
    for (auto& v : lg) v = rnd(4.0f);
    for (auto& v : bias) v = rnd(0.5f);
    float* dl = up(lg);
    float* db = up(bias);
    int* di = up(std::vector<int>((size_t) T * picks, -1));
    float* dw = up(std::vector<float>((size_t) T * picks, 0.0f));
    glm5::route(dl, ld, db, n, k, T, 2.5f, di, dw, picks, 0);
    const auto gi = down(di, (size_t) T * picks);
    const auto gw = down(dw, (size_t) T * picks);
    int bad = 0;
    double worst = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<int> o(n);
        for (int i = 0; i < n; ++i) o[i] = i;
        auto key = [&](int e) { return sig(lg[(size_t) t * ld + e]) + bias[e]; };
        std::sort(o.begin(), o.end(), [&](int a, int b) { return key(a) > key(b); });
        double sum = 0;
        for (int i = 0; i < k; ++i) sum += sig(lg[(size_t) t * ld + o[i]]);
        for (int i = 0; i < k; ++i) {
            bad += gi[(size_t) t * picks + i] != o[i];
            worst = std::max(worst, std::fabs(gw[(size_t) t * picks + i] - sig(lg[(size_t) t * ld + o[i]]) / sum * 2.5));
        }
    }
    const bool ok = bad == 0 && worst < 1e-5;
    std::printf("%-28s %d wrong picks, weights worst %.3e %s\n", "route", bad, worst, ok ? "ok" : "FAILED");
    failed += !ok;
}

int main() {
    test_kda();
    test_bmv();
    test_pool_attn();
    test_route();
    std::printf(failed ? "%d FAILED\n" : "ALL OK\n", failed);
    return failed != 0;
}
