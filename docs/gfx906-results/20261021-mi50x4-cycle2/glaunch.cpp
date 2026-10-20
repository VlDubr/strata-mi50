// host time of hipGraphLaunch: kernel-only graph vs graphs with a memcpy / memset / host-mapped-copy node
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
__global__ void k_small(float* p, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] = p[i] * 1.0001f + 1.0f; }
using Clk = std::chrono::steady_clock;
int main() {
    float *d, *d2, *h; hipMalloc(&d, 1 << 22); hipMalloc(&d2, 1 << 22); hipHostMalloc(&h, 1 << 20, hipHostMallocMapped);
    hipStream_t s; hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    const char* nm[] = {"400 kernels", "+1 D2D memcpy", "+1 H2D memcpy (pinned)", "+1 memset", "+10 D2D memcpy"};
    for (int v = 0; v < 5; ++v) {
        hipGraph_t g; hipGraphExec_t ge;
        hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal);
        for (int i = 0; i < 400; ++i) {
            k_small<<<10, 256, 0, s>>>(d, 2560);
            if (v == 1 && i == 200) hipMemcpyAsync(d2, d, 4096, hipMemcpyDeviceToDevice, s);
            if (v == 2 && i == 200) hipMemcpyAsync(d2, h, 4096, hipMemcpyHostToDevice, s);
            if (v == 3 && i == 200) hipMemsetAsync(d2, 0, 4096, s);
            if (v == 4 && i % 40 == 0) hipMemcpyAsync(d2, d, 4096, hipMemcpyDeviceToDevice, s);
        }
        hipStreamEndCapture(s, &g);
        hipGraphInstantiate(&ge, g, nullptr, nullptr, 0);
        hipGraphLaunch(ge, s); hipStreamSynchronize(s);
        double host = 0, total = 0;
        for (int r = 0; r < 20; ++r) {
            auto t0 = Clk::now(); hipGraphLaunch(ge, s); auto t1 = Clk::now(); hipStreamSynchronize(s); auto t2 = Clk::now();
            host += std::chrono::duration<double, std::micro>(t1 - t0).count();
            total += std::chrono::duration<double, std::micro>(t2 - t0).count();
        }
        std::printf("%-24s launch host %.1f us, launch->done %.1f us\n", nm[v], host / 20, total / 20);
    }
    return 0;
}
