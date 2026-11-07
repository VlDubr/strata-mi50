// GLM-5.3's routed experts on the device (Q4_K gate / up, Q5_K down), on made-up weights: the kernels' rate and a hash of what they
// write, to compare a change with the kernels before it.  usage: glm5_moe_bench [rows] [experts]; needs a GPU, no model.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../../src/dsv41/ds4_iq.cuh"

using namespace strata::ds4::gguf;

#define CK(x)                                                                               \
    do {                                                                                    \
        cudaError_t e_ = (x);                                                               \
        if (e_ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(2);                                                                   \
        }                                                                                   \
    } while (0)

static void fill_blocks(uint8_t* p, size_t blocks, size_t bytes, std::mt19937& rng) {
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* q = p + b * bytes;
        for (size_t i = 0; i < bytes; ++i) q[i] = (uint8_t) rng();
        q[0] = 0x00, q[1] = 0x18;   // d, dmin: small F16s
        q[2] = 0x00, q[3] = 0x14;
    }
}
template <class V>
static uint64_t hash_of(const V& v) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* p = (const uint8_t*) v.data();
    for (size_t i = 0; i < v.size() * sizeof(v[0]); ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

int main(int ac, char** av) {
    const int T = ac > 1 ? std::atoi(av[1]) : 3, G = ac > 2 ? std::atoi(av[2]) : 14;
    const int E = 4096, F = 2048;
    const size_t gr = (size_t) E / 256 * 144, dr = (size_t) F / 256 * 176;
    const size_t mg = (size_t) F * gr, md = (size_t) E * dr, blob = 2 * mg + md;
    std::mt19937 rng(7);
    std::vector<uint8_t> hb(blob);
    uint8_t* dw = nullptr;
    const int copies = 8;   // (more weights than the cache holds)
    CK(cudaMalloc(&dw, blob * G * copies));
    for (int g = 0; g < G * copies; ++g) {
        if (g < 4) {
            fill_blocks(hb.data(), 2 * mg / 144, 144, rng);
            fill_blocks(hb.data() + 2 * mg, md / 176, 176, rng);
        }
        CK(cudaMemcpy(dw + (size_t) g * blob, hb.data(), blob, cudaMemcpyHostToDevice));
    }
    // each group's entries: rows of the window, in turn (a row's entries are 8 apart: e = t * 8 + pick)
    std::vector<int> start(1, 0), ent;
    for (int g = 0; g < G; ++g) {
        const int ne = 1 + g % T;
        for (int i = 0; i < ne; ++i) ent.push_back(((g + i) % T) * 8 + g % 8);
        start.push_back((int) ent.size());
    }
    int8_t *xq, *hq;
    float *xd, *hd, *h, *part;
    int *dstart, *dent, *dng;
    unsigned long long* dptr;
    std::vector<int8_t> q((size_t) 64 * E);
    std::vector<float> sc((size_t) 64 * E / 32);
    for (auto& v : q) v = (int8_t) rng();
    for (auto& v : sc) v = 0.01f + (float) (rng() % 100) * 1e-4f;
    CK(cudaMalloc(&xq, q.size()));
    CK(cudaMalloc(&hq, q.size()));
    CK(cudaMalloc(&xd, sc.size() * 4));
    CK(cudaMalloc(&hd, sc.size() * 4));
    CK(cudaMalloc(&h, (size_t) 64 * F * 4));
    CK(cudaMalloc(&part, (size_t) 64 * E * 4));
    CK(cudaMemcpy(xq, q.data(), q.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(hq, q.data(), q.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xd, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(hd, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dstart, start.size() * 4));
    CK(cudaMalloc(&dent, ent.size() * 4));
    CK(cudaMalloc(&dng, 4));
    CK(cudaMalloc(&dptr, (size_t) G * copies * 8));
    CK(cudaMemcpy(dstart, start.data(), start.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dent, ent.data(), ent.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dng, &G, 4, cudaMemcpyHostToDevice));
    std::vector<unsigned long long> ptr((size_t) G * copies);
    for (size_t i = 0; i < ptr.size(); ++i) ptr[i] = (unsigned long long) (dw + i * blob);
    CK(cudaMemcpy(dptr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice));
    iq::Args a;
    a.grp_start = dstart;
    a.grp_ent = dent;
    a.n_groups = dng;
    a.tr[0] = 0;
    a.tr[1] = mg;
    a.tr[2] = 2 * mg;
    a.n_embd = E;
    a.n_ff = F;
    a.xq = xq;
    a.xd = xd;
    a.hq = hq;
    a.hd = hd;
    a.h = h;
    a.part = part;
    a.tg = iq::kQ4_K;
    a.td = iq::kQ5_K;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int reps = 40;
    float gu = 0, dn = 0;
    a.grp_ptr = dptr;
    iq::gate_up(a, G, T, 0);   // (the first launches load the code)
    iq::down(a, G, T, 0);
    CK(cudaDeviceSynchronize());
    for (int pass = 0; pass < 2; ++pass) {
        CK(cudaEventRecord(e0));
        for (int r = 0; r < reps; ++r) {
            a.grp_ptr = dptr + (size_t) (r % copies) * G;
            if (!(pass == 0 ? iq::gate_up(a, G, T, 0) : iq::down(a, G, T, 0))) {
                std::fprintf(stderr, "not covered\n");
                return 1;
            }
        }
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(pass == 0 ? &gu : &dn, e0, e1));
    }
    a.grp_ptr = dptr;
    CK(cudaMemset(h, 0, (size_t) 64 * F * 4));
    CK(cudaMemset(part, 0, (size_t) 64 * E * 4));
    iq::gate_up(a, G, T, 0);
    iq::down(a, G, T, 0);
    std::vector<float> oh((size_t) 64 * F), op((size_t) 64 * E);
    CK(cudaMemcpy(oh.data(), h, oh.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(op.data(), part, op.size() * 4, cudaMemcpyDeviceToHost));
    std::printf("rows %d experts %d: gate_up %.1f us (%.0f GB/s), down %.1f us (%.0f GB/s), hashes %016llx %016llx\n", T, G, gu * 1000 / reps,
                2.0 * mg * G * reps / (gu * 1e-3) / 1e9, dn * 1000 / reps, (double) md * G * reps / (dn * 1e-3) / 1e9,
                (unsigned long long) hash_of(oh), (unsigned long long) hash_of(op));
    return 0;
}
