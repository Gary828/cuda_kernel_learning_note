#include <cuda_runtime.h>
#include <cstdio>

// static inline void check_cuda(cudaError_t err, const char* what) {
//     if (err != cudaSuccess) {
//         std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
//     }
// }

__global__ void silu_kernel(const float* input, float* output, int N){
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < N){
        float value = input[gid];
        output[gid] = value / (1 + exp2f(-value * 1.4426950408889634f));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    // check_cuda(cudaGetLastError(), "silu_kernel launch");
    // check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaDeviceSynchronize();
}
