#include <cuda_runtime.h>
#include <cfloat>

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

__global__ void softmax_online_two_pass_kernel(const float* input, float* output, int N) {
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

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;
    softmax_online_two_pass_kernel<<<1, BLOCK_SIZE>>>(input, output, N);
    cudaDeviceSynchronize();
}
