#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void k_empty(float* p) { if (threadIdx.x == 1000000) p[0] = 1; }
__global__ void k_small(float* p, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] = p[i] * 1.0001f + 1.0f; }
int main() {
    float* d; hipMalloc(&d, 1 << 24);
    hipStream_t s; hipStreamCreate(&s);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    const int N = 400;
    for (int variant = 0; variant < 4; ++variant) {
        hipGraph_t g; hipGraphExec_t ge;
        hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal);
        for (int i = 0; i < N; ++i) {
            if (variant == 0) k_empty<<<1, 64, 0, s>>>(d);
            else if (variant == 1) k_small<<<10, 256, 0, s>>>(d, 2560);
            else if (variant == 2) k_small<<<160, 256, 0, s>>>(d, 40960);
            else k_small<<<2560, 256, 0, s>>>(d, 655360);
        }
        hipStreamEndCapture(s, &g);
        hipGraphInstantiate(&ge, g, nullptr, nullptr, 0);
        for (int w = 0; w < 3; ++w) hipGraphLaunch(ge, s);
        hipStreamSynchronize(s);
        float best = 1e9;
        for (int r = 0; r < 5; ++r) {
            hipEventRecord(a, s); hipGraphLaunch(ge, s); hipEventRecord(b, s); hipEventSynchronize(b);
            float ms; hipEventElapsedTime(&ms, a, b); if (ms < best) best = ms;
        }
        // same in a plain stream
        float best2 = 1e9;
        for (int r = 0; r < 5; ++r) {
            hipEventRecord(a, s);
            for (int i = 0; i < N; ++i) {
                if (variant == 0) k_empty<<<1, 64, 0, s>>>(d);
                else if (variant == 1) k_small<<<10, 256, 0, s>>>(d, 2560);
                else if (variant == 2) k_small<<<160, 256, 0, s>>>(d, 40960);
                else k_small<<<2560, 256, 0, s>>>(d, 655360);
            }
            hipEventRecord(b, s); hipEventSynchronize(b);
            float ms; hipEventElapsedTime(&ms, a, b); if (ms < best2) best2 = ms;
        }
        const char* nm[] = {"empty 1x64", "small 10x256", "small 160x256", "small 2560x256"};
        std::printf("%-16s graph %.2f us/kernel | stream %.2f us/kernel\n", nm[variant], best * 1000 / N, best2 * 1000 / N);
        hipGraphExecDestroy(ge); hipGraphDestroy(g);
    }
    return 0;
}
