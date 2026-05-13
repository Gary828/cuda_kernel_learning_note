#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float warp_max[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;

    v = warp_reduce_max(v);
    if (lane == 0) warp_max[warp_id] = v;
    __syncthreads();

    float out = -FLT_MAX;
    if (warp_id == 0) {
        out = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_max[lane] : -FLT_MAX;
        out = warp_reduce_max(out);
        if (lane == 0) warp_max[0] = out;
    }
    __syncthreads();
    return warp_max[0];
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_sum[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;

    v = warp_reduce_sum(v);
    if (lane == 0) warp_sum[warp_id] = v;
    __syncthreads();

    float out = 0.0f;
    if (warp_id == 0) {
        out = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_sum[lane] : 0.0f;
        out = warp_reduce_sum(out);
        if (lane == 0) warp_sum[0] = out;
    }
    __syncthreads();
    return warp_sum[0];
}

__global__ void softmax_scalar_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;

    float local_max = -FLT_MAX;
    for (int col = tid; col < N; col += blockDim.x) {
        local_max = fmaxf(local_max, input[col]);
    }
    float row_max = block_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        local_sum += __expf(input[col] - row_max);
    }
    float row_sum = block_reduce_sum(local_sum);
    float inv_sum = 1.0f / row_sum;

    for (int col = tid; col < N; col += blockDim.x) {
        output[col] = __expf(input[col] - row_max) * inv_sum;
    }
}

__global__ void softmax_float4_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int N_vec4 = N / 4;
    int tail_start = N_vec4 * 4;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float local_max = -FLT_MAX;
    for (int i = tid; i < N_vec4; i += blockDim.x) {
        float4 v = input4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    for (int col = tail_start + tid; col < N; col += blockDim.x) {
        local_max = fmaxf(local_max, input[col]);
    }
    float row_max = block_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int i = tid; i < N_vec4; i += blockDim.x) {
        float4 v = input4[i];
        local_sum += __expf(v.x - row_max);
        local_sum += __expf(v.y - row_max);
        local_sum += __expf(v.z - row_max);
        local_sum += __expf(v.w - row_max);
    }
    for (int col = tail_start + tid; col < N; col += blockDim.x) {
        local_sum += __expf(input[col] - row_max);
    }
    float row_sum = block_reduce_sum(local_sum);
    float inv_sum = 1.0f / row_sum;

    float4* output4 = reinterpret_cast<float4*>(output);
    for (int i = tid; i < N_vec4; i += blockDim.x) {
        float4 v = input4[i];
        float4 out;
        out.x = __expf(v.x - row_max) * inv_sum;
        out.y = __expf(v.y - row_max) * inv_sum;
        out.z = __expf(v.z - row_max) * inv_sum;
        out.w = __expf(v.w - row_max) * inv_sum;
        output4[i] = out;
    }
    for (int col = tail_start + tid; col < N; col += blockDim.x) {
        output[col] = __expf(input[col] - row_max) * inv_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    bool aligned =
        (reinterpret_cast<uintptr_t>(input) % alignof(float4) == 0) &&
        (reinterpret_cast<uintptr_t>(output) % alignof(float4) == 0);

    if (aligned && N >= 4) {
        softmax_float4_kernel<<<1, BLOCK_SIZE>>>(input, output, N);
    } else {
        softmax_scalar_kernel<<<1, BLOCK_SIZE>>>(input, output, N);
    }
    cudaDeviceSynchronize();
}
