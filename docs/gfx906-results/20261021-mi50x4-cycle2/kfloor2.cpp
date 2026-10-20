#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void k_small(float* p, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] = p[i] * 1.0001f + 1.0f; }
struct Big { float* p[160]; int n; };
__global__ void k_bigarg(Big b) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < b.n) b.p[0][i] = b.p[0][i] * 1.0001f + 1.0f; }
int main() {
    float *d, *d2, *h; hipMalloc(&d, 1 << 24); hipMalloc(&d2, 1 << 24);
    hipHostMalloc(&h, 1 << 20, hipHostMallocMapped);
    hipStream_t s, s2; hipStreamCreate(&s); hipStreamCreate(&s2);
    hipEvent_t a, b, ef, ej; hipEventCreate(&a); hipEventCreate(&b);
    hipEventCreateWithFlags(&ef, hipEventDisableTiming); hipEventCreateWithFlags(&ej, hipEventDisableTiming);
    const int N = 400;
    Big big{}; big.p[0] = d; big.n = 2560;
    const char* nm[] = {"kernels only", "+1 memcpy D2D node", "+1 fork/join", "big kernarg (1.3KB)", "+memcpy every 20", "+memsetAsync node"};
    for (int variant = 0; variant < 6; ++variant) {
        hipGraph_t g; hipGraphExec_t ge;
        hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal);
        for (int i = 0; i < N; ++i) {
            if (variant == 3) k_bigarg<<<10, 256, 0, s>>>(big);
            else k_small<<<10, 256, 0, s>>>(d, 2560);
            if (variant == 1 && i == N / 2) hipMemcpyAsync(d2, d, 4096, hipMemcpyDeviceToDevice, s);
            if (variant == 4 && i % 20 == 0) hipMemcpyAsync(d2, d, 4096, hipMemcpyDeviceToDevice, s);
            if (variant == 5 && i == N / 2) hipMemsetAsync(d2, 0, 4096, s);
            if (variant == 2 && i == N / 2) {
                hipEventRecord(ef, s); hipStreamWaitEvent(s2, ef, 0);
                k_small<<<10, 256, 0, s2>>>(d2, 2560);
                hipEventRecord(ej, s2); hipStreamWaitEvent(s, ej, 0);
            }
        }
        hipStreamEndCapture(s, &g);
        size_t nn = 0; hipGraphGetNodes(g, nullptr, &nn);
        hipGraphInstantiate(&ge, g, nullptr, nullptr, 0);
        for (int w = 0; w < 3; ++w) hipGraphLaunch(ge, s);
        hipStreamSynchronize(s);
        float best = 1e9;
        for (int r = 0; r < 5; ++r) {
            hipEventRecord(a, s); hipGraphLaunch(ge, s); hipEventRecord(b, s); hipEventSynchronize(b);
            float ms; hipEventElapsedTime(&ms, a, b); if (ms < best) best = ms;
        }
        std::printf("%-22s nodes %4zu graph %.2f us/kernel\n", nm[variant], nn, best * 1000 / N);
        hipGraphExecDestroy(ge); hipGraphDestroy(g);
    }
    return 0;
}
