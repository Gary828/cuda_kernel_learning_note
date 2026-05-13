#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr int kItemsPerBlock = 2 * kThreads;

// 每个 block 处理 2 * kThreads 个元素。
// 这里实现的是：
// 1. 先在 shared memory 里做一轮块内 Blelloch scan
// 2. 顺手把当前 block 的总和写到 block_sums[blockIdx.x]
// 3. 最后把 exclusive scan 转成 inclusive scan 后写回 output
__global__ void scan_block_kernel(const float* input,
                                  float* output,
                                  float* block_sums,
                                  int N) {
    __shared__ float temp[kItemsPerBlock];

    const int tid = threadIdx.x;
    const int base = blockIdx.x * kItemsPerBlock;
    const int left = base + tid;
    const int right = left + kThreads;

    // 一个线程负责两个位置：
    // temp[tid]           对应 block 前半段
    // temp[tid + kThreads] 对应 block 后半段
    // 越界位置补 0，不影响 scan 正确性。
    temp[tid] = (left < N) ? input[left] : 0.0f;
    temp[tid + kThreads] = (right < N) ? input[right] : 0.0f;
    __syncthreads();

    // Blelloch upsweep：
    // 从底向上构造一棵求和树，最终 temp[kItemsPerBlock - 1]
    // 会变成整个 block 的总和。
    for (int offset = 1; offset < kItemsPerBlock; offset <<= 1) {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < kItemsPerBlock) {
            temp[idx] += temp[idx - offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // 记录当前 block 的元素总和，后面跨 block 时要用。
        block_sums[blockIdx.x] = temp[kItemsPerBlock - 1];

        // 把根节点置 0，准备做 downsweep。
        // 这一步之后，树会被改造成 exclusive scan。
        temp[kItemsPerBlock - 1] = 0.0f;
    }
    __syncthreads();

    // Blelloch downsweep：
    // 把刚才那棵“求总和的树”改造成“前缀和的树”。
    // 做完之后，temp 里保存的是 exclusive scan 结果。
    for (int offset = kItemsPerBlock >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < kItemsPerBlock) {
            float val = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += val;
        }
        __syncthreads();
    }

    // downsweep 得到的是 exclusive scan：
    // temp[pos] = input[0..pos-1] 的和
    // 这里题目更常见的是 inclusive scan，所以把当前位置原值加回去。
    if (left < N) {
        output[left] = temp[tid] + input[left];
    }
    if (right < N) {
        output[right] = temp[tid + kThreads] + input[right];
    }
}

// 当前 block 的局部 scan 已经对了，
// 但 block 之间还缺“前面所有 block 的总和”这个偏移量。
// scanned_block_sums 里保存的就是每个 block 对应的前缀块和。
__global__ void add_offsets_kernel(float* output, const float* scanned_block_sums, int N) {
    const int tid = threadIdx.x;
    const int base = blockIdx.x * kItemsPerBlock;
    const int left = base + tid;
    const int right = left + kThreads;
    const float offset = (blockIdx.x == 0) ? 0.0f : scanned_block_sums[blockIdx.x - 1];

    if (left < N) {
        output[left] += offset;
    }
    if (right < N) {
        output[right] += offset;
    }
}

// 递归版 scan：
// 1. 先做 block 内 scan
// 2. 再对 block_sums 自己做 scan
// 3. 把每个 block 该加的偏移量回填到 output
void scan_recursive(const float* input, float* output, int N) {
    if (N <= 0) {
        return;
    }

    const int blocks = (N + kItemsPerBlock - 1) / kItemsPerBlock;

    float* block_sums = nullptr;
    cudaMalloc(&block_sums, blocks * sizeof(float));

    // 第一步：每个 block 先独立完成局部前缀和。
    scan_block_kernel<<<blocks, kThreads>>>(input, output, block_sums, N);

    if (blocks > 1) {
        float* scanned = nullptr;
        cudaMalloc(&scanned, blocks * sizeof(float));

        // 第二步：block_sums 本身还是个数组，所以继续递归做 scan。
        scan_recursive(block_sums, scanned, blocks);

        // 第三步：把“前面所有 block 的总和”加回当前 block。
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

    // 对外暴露的是 inclusive scan：
    // output[i] = input[0] + input[1] + ... + input[i]
    scan_recursive(input, output, N);
    cudaDeviceSynchronize();
}
