#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

struct __align__(8) OnlineSoftmaxState {
    float max_val;
    float sum;
};

__device__ __forceinline__ OnlineSoftmaxState make_online_state(float max_val, float sum) {
    OnlineSoftmaxState v;
    v.max_val = max_val;
    v.sum = sum;
    return v;
}

__device__ __forceinline__ OnlineSoftmaxState online_merge(OnlineSoftmaxState a, OnlineSoftmaxState b) {
    float new_max = fmaxf(a.max_val, b.max_val);
    float new_sum =
        a.sum * __expf(a.max_val - new_max) +
        b.sum * __expf(b.max_val - new_max);
    return make_online_state(new_max, new_sum);
}

template <int kWarpWidth>
__device__ __forceinline__ OnlineSoftmaxState warp_reduce_online(OnlineSoftmaxState v) {
    #pragma unroll
    for (int mask = kWarpWidth >> 1; mask > 0; mask >>= 1) {
        float other_max = __shfl_xor_sync(0xffffffff, v.max_val, mask);
        float other_sum = __shfl_xor_sync(0xffffffff, v.sum, mask);
        v = online_merge(v, make_online_state(other_max, other_sum));
    }
    return v;
}

__device__ __forceinline__ OnlineSoftmaxState block_reduce_online(OnlineSoftmaxState v) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ OnlineSoftmaxState warp_state[NUM_WARPS];

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;

    v = warp_reduce_online<WARP_SIZE>(v);
    if (lane == 0) warp_state[warp_id] = v;
    __syncthreads();

    if (warp_id == 0) {
        v = (lane < NUM_WARPS) ? warp_state[lane] : make_online_state(-FLT_MAX, 0.0f);
        v = warp_reduce_online<NUM_WARPS>(v);
        if (lane == 0) warp_state[0] = v;
    }
    __syncthreads();
    return warp_state[0];
}

__global__ void softmax_online_scalar_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;

    float local_max = -FLT_MAX;
    for (int col = tid; col < N; col += blockDim.x) {
        local_max = fmaxf(local_max, input[col]);
    }

    float local_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        local_sum += __expf(input[col] - local_max);
    }

    OnlineSoftmaxState row_state = block_reduce_online(make_online_state(local_max, local_sum));
    float inv_sum = 1.0f / row_state.sum;

    for (int col = tid; col < N; col += blockDim.x) {
        output[col] = __expf(input[col] - row_state.max_val) * inv_sum;
    }
}

__global__ void softmax_online_float4_kernel(const float* input, float* output, int N) {
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

    float local_sum = 0.0f;
    for (int i = tid; i < N_vec4; i += blockDim.x) {
        float4 v = input4[i];
        local_sum += __expf(v.x - local_max);
        local_sum += __expf(v.y - local_max);
        local_sum += __expf(v.z - local_max);
        local_sum += __expf(v.w - local_max);
    }
    for (int col = tail_start + tid; col < N; col += blockDim.x) {
        local_sum += __expf(input[col] - local_max);
    }

    OnlineSoftmaxState row_state = block_reduce_online(make_online_state(local_max, local_sum));
    float inv_sum = 1.0f / row_state.sum;

    float4* output4 = reinterpret_cast<float4*>(output);
    for (int i = tid; i < N_vec4; i += blockDim.x) {
        float4 v = input4[i];
        float4 out;
        out.x = __expf(v.x - row_state.max_val) * inv_sum;
        out.y = __expf(v.y - row_state.max_val) * inv_sum;
        out.z = __expf(v.z - row_state.max_val) * inv_sum;
        out.w = __expf(v.w - row_state.max_val) * inv_sum;
        output4[i] = out;
    }
    for (int col = tail_start + tid; col < N; col += blockDim.x) {
        output[col] = __expf(input[col] - row_state.max_val) * inv_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    bool aligned =
        (reinterpret_cast<uintptr_t>(input) % alignof(float4) == 0) &&
        (reinterpret_cast<uintptr_t>(output) % alignof(float4) == 0);

    if (aligned && N >= 4) {
        softmax_online_float4_kernel<<<1, BLOCK_SIZE>>>(input, output, N);
    } else {
        softmax_online_scalar_kernel<<<1, BLOCK_SIZE>>>(input, output, N);
    }
    cudaDeviceSynchronize();
}
