#include <cuda_runtime.h>


// FP32 x4: one thread handles 4 float elements
__global__ void elementwise_add_fp32x4_kernel(const float* a, const float* b, float* c, int N){
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx >= N)return;
    if(idx + 3 < N){
        float4 av = *reinterpret_cast<const float4 *>(a + idx);
        float4 bv = *reinterpret_cast<const float4 *>(b + idx);
        float4 cv = make_float4(av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w);
        *reinterpret_cast<float4 *>(c + idx) = cv;
    } else {
        if(idx + 0 < N)c[idx + 0] = a[idx + 0] + b[idx + 0];
        if(idx + 1 < N)c[idx + 1] = a[idx + 1] + b[idx + 1];
        if(idx + 2 < N)c[idx + 2] = a[idx + 2] + b[idx + 2];
        // if(idx + 3 < N)c[idx + 3] = a[idx + 3] + b[idx + 3];
    }
}


extern "C" void solve_fp32x4(const float *a, const float *b, float *c, int N) {
    if (N <= 0) return;
    int threads = 256;
    int blocks = (N + 4 * threads - 1) / (4 * threads);
    elementwise_add_fp32x4_kernel<<<blocks, threads>>>(a, b, c, N);
}
