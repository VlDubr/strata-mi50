// parity + timing: route / route_multi (32-lane) vs route_wave64 (gfx906).  Build from this folder:
//   hipcc -O3 --offload-arch=gfx906 -x hip -std=c++17 -DSTRATA_HIP_GFX906=1 -D__HIP_PLATFORM_AMD__=1
//     -I../../../include/strata/platform/hip_compat -I../../../include route_test.cpp -o route_test
#include "../../../src/kernels/cuda/native_router.cu"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
using namespace strata::kernels;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s: %s\n", #x, hipGetErrorString(e_)); return 1; } } while (0)
int main(int argc, char** argv) {
    const int rows = 1 << 16;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::vector<float> h((size_t) rows * 512);
    for (int r = 0; r < rows; ++r) {
        const int mode = r % 8;
        for (int e = 0; e < 512; ++e) {
            float x = nd(rng);
            if (mode == 1) x = std::round(x * 2.0f) / 2.0f;              // many exact ties
            if (mode == 2) x = (float) (int) (x) ;                        // heavy ties
            if (mode == 3 && (e % 37) == 0) x = NAN;
            if (mode == 4) x = 0.0f;                                      // all equal
            if (mode == 5 && e < 20) x = 50.0f;                           // 20-way tie at the top
            if (mode == 6) x = x * 30.0f;                                 // underflow most
            if (mode == 7 && (e % 3) == 0) x = -INFINITY;
            h[(size_t) r * 512 + e] = x;
        }
    }
    float *d_l, *d_w1, *d_w2; int32_t *d_i1, *d_i2;
    CK(hipMalloc(&d_l, h.size() * 4));
    CK(hipMalloc(&d_w1, (size_t) rows * 10 * 4)); CK(hipMalloc(&d_w2, (size_t) rows * 10 * 4));
    CK(hipMalloc(&d_i1, (size_t) rows * 10 * 4)); CK(hipMalloc(&d_i2, (size_t) rows * 10 * 4));
    CK(hipMemcpy(d_l, h.data(), h.size() * 4, hipMemcpyHostToDevice));
    route_multi<<<(rows + 7) / 8, dim3(32, 8)>>>(d_l, d_i1, d_w1, rows);
    route_wave64<<<rows, 64>>>(d_l, d_i2, d_w2);
    CK(hipDeviceSynchronize());
    std::vector<int32_t> i1((size_t) rows * 10), i2((size_t) rows * 10);
    std::vector<uint32_t> w1((size_t) rows * 10), w2((size_t) rows * 10);
    CK(hipMemcpy(i1.data(), d_i1, i1.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(i2.data(), d_i2, i2.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(w1.data(), d_w1, w1.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(w2.data(), d_w2, w2.size() * 4, hipMemcpyDeviceToHost));
    long bad_i = 0, bad_w = 0;
    for (size_t k = 0; k < i1.size(); ++k) { bad_i += i1[k] != i2[k]; bad_w += w1[k] != w2[k]; }
    // single-token kernel `route` vs wave64 on the first 4096 rows
    long bad_s = 0;
    for (int r = 0; r < 4096; ++r) {
        route<<<1, dim3(32, 8)>>>(d_l + (size_t) r * 512, d_i1 + (size_t) r * 10, d_w1 + (size_t) r * 10);
    }
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(i1.data(), d_i1, 4096 * 40, hipMemcpyDeviceToHost));
    CK(hipMemcpy(w1.data(), d_w1, 4096 * 40, hipMemcpyDeviceToHost));
    for (size_t k = 0; k < 4096 * 10; ++k) bad_s += (i1[k] != i2[k]) || (w1[k] != w2[k]);
    std::printf("parity: %d rows, ids differing %ld, weights differing %ld, single-token route differing %ld\n", rows, bad_i, bad_w, bad_s);
    // timing: dependent chain of single-token launches (as in a graph: one per layer)
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int rep = 0; rep < 2; ++rep) {
        const int N = 2000;
        CK(hipEventRecord(a));
        for (int k = 0; k < N; ++k) route<<<1, dim3(32, 8)>>>(d_l + (size_t) (k % 64) * 512, d_i1, d_w1);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms1; CK(hipEventElapsedTime(&ms1, a, b));
        CK(hipEventRecord(a));
        for (int k = 0; k < N; ++k) route_wave64<<<1, 64>>>(d_l + (size_t) (k % 64) * 512, d_i2, d_w2);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms2; CK(hipEventElapsedTime(&ms2, a, b));
        CK(hipEventRecord(a));
        for (int k = 0; k < N; ++k) route_multi<<<1, dim3(32, 8)>>>(d_l + (size_t) (k % 64) * 512, d_i1, d_w1, 4);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms3; CK(hipEventElapsedTime(&ms3, a, b));
        CK(hipEventRecord(a));
        for (int k = 0; k < N; ++k) route_wave64<<<4, 64>>>(d_l + (size_t) (k % 64) * 512, d_i2, d_w2);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms4; CK(hipEventElapsedTime(&ms4, a, b));
        std::printf("per launch (stream, incl. dispatch): route %.2f us, wave64 %.2f us | T=4: route_multi %.2f us, wave64 %.2f us\n",
                    ms1 * 1000 / N, ms2 * 1000 / N, ms3 * 1000 / N, ms4 * 1000 / N);
    }
    return (bad_i || bad_w || bad_s) ? 2 : 0;
}
