#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

// Kogge-Stone 块内 scan：
// 1. 每个 block 处理连续的 kThreads 个元素
// 2. 块内直接做 inclusive scan
// 3. 同时记录每个 block 的总和，供跨 block 偏移使用
__global__ void scan_block_kogge_stone_kernel(const float* input,
                                              float* output,
                                              float* block_sums,
                                              int N) {
    __shared__ float temp[kThreads];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    temp[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Kogge-Stone 的核心就是：
    // 第 1 轮看前 1 个
    // 第 2 轮看前 2 个
    // 第 3 轮看前 4 个
    // ...
    // 每一轮都把“更远距离的前缀信息”传播过来。
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float val = (tid >= offset) ? temp[tid - offset] : 0.0f;
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    if (idx < N) {
        output[idx] = temp[tid];
    }

    // Kogge-Stone 这里本身就是 inclusive scan，
    // 所以块内最后一个位置就是整个 block 的总和。
    if (tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = temp[tid];
    }
}

// 给每个 block 加上“前面所有 block 的总和”。
__global__ void add_offsets_kernel(float* output, const float* scanned_block_sums, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || blockIdx.x == 0) {
        return;
    }

    output[idx] += scanned_block_sums[blockIdx.x - 1];
}

// 递归扫描 block_sums：
// 如果 block_sums 也超过一个 block，就继续递归处理。
void scan_recursive_kogge_stone(const float* input, float* output, int N) {
    if (N <= 0) {
        return;
    }

    int blocks = (N + kThreads - 1) / kThreads;

    float* block_sums = nullptr;
    cudaMalloc(&block_sums, blocks * sizeof(float));

    scan_block_kogge_stone_kernel<<<blocks, kThreads>>>(input, output, block_sums, N);

    if (blocks > 1) {
        float* scanned = nullptr;
        cudaMalloc(&scanned, blocks * sizeof(float));

        scan_recursive_kogge_stone(block_sums, scanned, blocks);
        add_offsets_kernel<<<blocks, kThreads>>>(output, scanned, N);

        cudaFree(scanned);
    }

    cudaFree(block_sums);
}

}  // namespace

// input, output are device pointers.
extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) {
        return;
    }

    // 这版默认输出 inclusive scan：
    // output[i] = input[0] + ... + input[i]
    scan_recursive_kogge_stone(input, output, N);
    cudaDeviceSynchronize();
}
