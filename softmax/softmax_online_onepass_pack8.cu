#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int REG_PACKS = 8;
constexpr int PACK_SIZE = 4;
constexpr int MAX_CACHED_N = BLOCK_SIZE * REG_PACKS * PACK_SIZE;

struct __align__(8) OnlineSoftmaxState {
    float max_val;
    float sum;
};

union alignas(16) pack128 {
    float4 f4;
    float f[4];
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

__global__ void softmax_online_two_pass_fallback_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;

    OnlineSoftmaxState local = make_online_state(-FLT_MAX, 0.0f);
    for (int col = tid; col < N; col += blockDim.x) {
        local = online_merge(local, make_online_state(input[col], 1.0f));
    }

    OnlineSoftmaxState row_state = block_reduce_online(local);
    float inv_sum = 1.0f / row_state.sum;

    for (int col = tid; col < N; col += blockDim.x) {
        output[col] = __expf(input[col] - row_state.max_val) * inv_sum;
    }
}

__global__ void softmax_online_onepass_pack8_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    pack128 pack[REG_PACKS];
    float local_max = -FLT_MAX;

    #pragma unroll
    for (int i = 0; i < REG_PACKS; ++i) {
        int vec_idx = tid + i * BLOCK_SIZE;
        if (vec_idx * PACK_SIZE < N) {
            pack[i].f4 = input4[vec_idx];
            local_max = fmaxf(local_max, pack[i].f[0]);
            local_max = fmaxf(local_max, pack[i].f[1]);
            local_max = fmaxf(local_max, pack[i].f[2]);
            local_max = fmaxf(local_max, pack[i].f[3]);
        }
    }

    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < REG_PACKS; ++i) {
        int vec_idx = tid + i * BLOCK_SIZE;
        if (vec_idx * PACK_SIZE < N) {
            local_sum += __expf(pack[i].f[0] - local_max);
            local_sum += __expf(pack[i].f[1] - local_max);
            local_sum += __expf(pack[i].f[2] - local_max);
            local_sum += __expf(pack[i].f[3] - local_max);
        }
    }

    OnlineSoftmaxState row_state = block_reduce_online(make_online_state(local_max, local_sum));
    float inv_sum = 1.0f / row_state.sum;

    #pragma unroll
    for (int i = 0; i < REG_PACKS; ++i) {
        int vec_idx = tid + i * BLOCK_SIZE;
        if (vec_idx * PACK_SIZE < N) {
            pack128 out;
            out.f[0] = __expf(pack[i].f[0] - row_state.max_val) * inv_sum;
            out.f[1] = __expf(pack[i].f[1] - row_state.max_val) * inv_sum;
            out.f[2] = __expf(pack[i].f[2] - row_state.max_val) * inv_sum;
            out.f[3] = __expf(pack[i].f[3] - row_state.max_val) * inv_sum;
            output4[vec_idx] = out.f4;
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    bool aligned =
        (reinterpret_cast<uintptr_t>(input) % alignof(float4) == 0) &&
        (reinterpret_cast<uintptr_t>(output) % alignof(float4) == 0);
    bool use_onepass = aligned && (N % PACK_SIZE == 0) && (N <= MAX_CACHED_N);

    if (use_onepass) {
        softmax_online_onepass_pack8_kernel<<<1, BLOCK_SIZE>>>(input, output, N);
    } else {
        softmax_online_two_pass_fallback_kernel<<<1, BLOCK_SIZE>>>(input, output, N);
    }
    cudaDeviceSynchronize();
}
