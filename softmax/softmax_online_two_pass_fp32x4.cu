#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

struct MD {
    float m;
    float d;
};

__device__ __forceinline__ MD online_update(MD val, float x) {
    float new_m = fmaxf(val.m, x);
    val.d = val.d * __expf(val.m - new_m) + __expf(x - new_m);
    val.m = new_m;
    return val;
}

__device__ __forceinline__ MD online_merge(MD a, MD b) {
    float new_m = fmaxf(a.m, b.m);
    a.d = a.d * __expf(a.m - new_m) + b.d * __expf(b.m - new_m);
    a.m = new_m;
    return a;
}

__device__ __forceinline__ MD warp_reduce_online(MD val) {
    #pragma unroll
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(0xffffffff, val.m, mask);
        other.d = __shfl_xor_sync(0xffffffff, val.d, mask);
        val = online_merge(val, other);
    }
    return val;
}

__device__ __forceinline__ MD block_reduce_online(MD val) {
    __shared__ MD warp_vals[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_online(val);
    if (lane == 0) warp_vals[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_vals[lane] : MD{-FLT_MAX, 0.0f};
        val = warp_reduce_online(val);
        if (lane == 0) warp_vals[0] = val;
    }
    __syncthreads();
    return warp_vals[0];
}

__global__ void softmax_online_two_pass_fp32x4_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int vec_n = N / 4;
    int tail_start = vec_n * 4;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    MD val{-FLT_MAX, 0.0f};

    for (int i = tid; i < vec_n; i += BLOCK_SIZE) {
        float4 v = input4[i];
        val = online_update(val, v.x);
        val = online_update(val, v.y);
        val = online_update(val, v.z);
        val = online_update(val, v.w);
    }
    for (int i = tail_start + tid; i < N; i += BLOCK_SIZE) {
        val = online_update(val, input[i]);
    }

    val = block_reduce_online(val);
    float inv_d = 1.0f / val.d;

    for (int i = tid; i < vec_n; i += BLOCK_SIZE) {
        float4 v = input4[i];
        float4 out;
        out.x = __expf(v.x - val.m) * inv_d;
        out.y = __expf(v.y - val.m) * inv_d;
        out.z = __expf(v.z - val.m) * inv_d;
        out.w = __expf(v.w - val.m) * inv_d;
        output4[i] = out;
    }
    for (int i = tail_start + tid; i < N; i += BLOCK_SIZE) {
        output[i] = __expf(input[i] - val.m) * inv_d;
    }
}

__global__ void softmax_online_two_pass_scalar_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    MD val{-FLT_MAX, 0.0f};

    for (int i = tid; i < N; i += BLOCK_SIZE) {
        val = online_update(val, input[i]);
    }

    val = block_reduce_online(val);
    float inv_d = 1.0f / val.d;

    for (int i = tid; i < N; i += BLOCK_SIZE) {
        output[i] = __expf(input[i] - val.m) * inv_d;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    bool aligned =
        (reinterpret_cast<uintptr_t>(input) % alignof(float4) == 0) &&
        (reinterpret_cast<uintptr_t>(output) % alignof(float4) == 0);

    if (aligned && N >= 4) {
        softmax_online_two_pass_fp32x4_kernel<<<1, BLOCK_SIZE>>>(input, output, N);
    } else {
        softmax_online_two_pass_scalar_kernel<<<1, BLOCK_SIZE>>>(input, output, N);
    }
    cudaDeviceSynchronize();
}
