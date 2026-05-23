#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[kThreads / kWarpSize];

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp_id = threadIdx.x / kWarpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < kThreads / kWarpSize) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();

    return val;
}

__global__ void batchnorm_interview_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N,
    int C,
    float eps) {
    int c = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float mean;
    __shared__ float inv_std;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int n = tid; n < N; n += blockDim.x) {
        float x = input[n * C + c];
        sum += x;
        sumsq += x * x;
    }

    float block_sum = block_reduce_sum(sum);
    float block_sumsq = block_reduce_sum(sumsq);
    if (tid == 0) {
        mean = block_sum / static_cast<float>(N);
        float ex2 = block_sumsq / static_cast<float>(N);
        float var = fmaxf(ex2 - mean * mean, 0.0f);
        inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float g = gamma[c];
    float b = beta[c];
    for (int n = tid; n < N; n += blockDim.x) {
        int idx = n * C + c;
        float x_hat = (input[idx] - mean) * inv_std;
        output[idx] = g * x_hat + b;
    }
}

}  // namespace

// input, gamma, beta, output are device pointers.
extern "C" void solve(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int N,
    int C,
    float eps) {
    if (N <= 0 || C <= 0) {
        return;
    }

    batchnorm_interview_kernel<<<C, kThreads>>>(input, gamma, beta, output, N, C, eps);
    cudaDeviceSynchronize();
}
