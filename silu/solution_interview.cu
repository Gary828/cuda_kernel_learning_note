#include <cuda_runtime.h>
#include <cstdint>
__device__ __forceinline__ float sigmoid(float x){
    return 1.0f / (1 + __expf(-x));
}
// __global__ void silu_kernel(const float* input, float* output, int N) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     if(idx >= N)return;
//     float x = input[idx];
//     output[idx] = x / ( 1 + expf(-x));
// }

__global__ void silu_scalar_kernel(const float* input, float* output, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < N; i += stride){
        float x = input[i];
        output[i] = x * sigmoid(x);
    }
}

__global__ void silu_float4_kernel(const float* input, float* output, int N){
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_stride = blockDim.x * gridDim.x;
    int N_vec4 = N / 4;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    for(int i = vec_idx; i < N_vec4; i += vec_stride){
        float4 x4 = input4[i];
        float4 y4;
        y4.x = x4.x * sigmoid(x4.x);
        y4.y = x4.y * sigmoid(x4.y);
        y4.z = x4.z * sigmoid(x4.z);
        y4.w = x4.w * sigmoid(x4.w);
        output4[i] = y4;
    }

    int tail_start = N_vec4 * 4;
    for(int i = tail_start + vec_idx; i < N; i += vec_stride){
        float x = input[i];
        output[i] = x * sigmoid(x);
    }
}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    if(N <= 0)return;
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    bool aligned = 
        (reinterpret_cast<uintptr_t>(input) % alignof(float4) == 0) &&
        (reinterpret_cast<uintptr_t>(output) % alignof(float4) == 0);
    if(aligned && N >= 4){
        int vec_blocks = ((N / 4) + threadsPerBlock - 1) / threadsPerBlock;
        silu_float4_kernel<<<vec_blocks, threadsPerBlock>>>(input, output, N);
    } else {
        silu_scalar_kernel<<<blocks, threadsPerBlock>>>(input, output, N);
    }
    cudaDeviceSynchronize();
}
