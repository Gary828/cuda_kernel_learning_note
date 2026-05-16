#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr int kThreads = 256;

// 向上取整除法。
// 例如 div_up(1000, 256) = 4，表示需要 4 个 block 才能覆盖 1000 个元素。
__host__ __device__ inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// 经典 bitonic sort 一般要求输入长度是 2 的幂。
// 如果原始长度 N 不是 2 的幂，这里就把它补到 >= N 的最小 2 的幂。
inline int next_pow2(int n) {
    int p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

// 第一步：把原始输入拷到临时数组 data。
// 如果 padded_n > n，多出来的位置统一补成 -INF。
//
// 为什么补 -INF？
// 因为这题要找的是 top-k 最大值。
// 补进去的值必须保证永远不可能进入最终答案，所以选负无穷最直接。
__global__ void copy_and_pad_kernel(
    const float* input,
    float* data,
    int n,
    int padded_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= padded_n) {
        return;
    }
    data[idx] = (idx < n) ? input[idx] : -CUDART_INF_F;
}

// Bitonic sort 的单步 compare-swap kernel。
//
// 参数含义：
// - stage：当前 bitonic merge 的大阶段，决定当前是在多大的子序列内做排序
// - stride：当前这一轮比较的间隔距离
//
// 例如：
// - stage = 8, stride = 4：表示在长度为 8 的子序列里，先比较相距 4 的元素
// - 然后 stride = 2，再 stride = 1，逐渐把这段子序列完全排好
__global__ void bitonic_step_kernel(float* data, int stride, int stage, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= static_cast<unsigned int>(n)) {
        return;
    }

    // XOR 配对是 bitonic network 的经典写法。
    // 它的含义是：当前 idx 在这一轮应该和谁比较。
    //
    // 例如 stride = 1：
    // 0 <-> 1, 2 <-> 3, 4 <-> 5 ...
    //
    // 例如 stride = 2：
    // 0 <-> 2, 1 <-> 3, 4 <-> 6 ...
    unsigned int peer = idx ^ stride;

    // peer <= idx：避免同一对元素被处理两次
    // peer >= n：越界保护
    if (peer <= idx || peer >= static_cast<unsigned int>(n)) {
        return;
    }

    float a = data[idx];
    float b = data[peer];

    // 当前这一对元素应该按升序还是降序比较，
    // 由 idx 所在 stage 的那一位决定。
    //
    // 可以把 stage 看成“当前要构造的 bitonic 段长度”。
    // 当 (idx & stage) == 0 时，这一半按升序方向整理；
    // 否则按降序方向整理。
    bool ascending = ((idx & stage) == 0);

    // 如果当前顺序不符合要求，就交换。
    // 升序：前面的值应该 <= 后面的值
    // 降序：前面的值应该 >= 后面的值
    if ((ascending && a > b) || (!ascending && a < b)) {
        data[idx] = b;
        data[peer] = a;
    }
}

// 所有 bitonic 排序阶段完成后，data 已经是全局升序。
// 因此最后 k 个元素就是最大的 k 个值。
// 这里从尾部倒着取出来，写到 output[0..k-1]。
__global__ void gather_topk_kernel(
    const float* data,
    float* output,
    int padded_n,
    int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) {
        return;
    }

    // Bitonic 网络按升序排好后，最后 k 个就是最大的 k 个。
    output[idx] = data[padded_n - 1 - idx];
}

}  // namespace

// input, output are device pointers
//
// 整体流程：
// 1. 把输入补齐到 2 的幂
// 2. 对补齐后的数组做全局 bitonic sort
// 3. 排序结果是升序，所以从尾部取前 k 个最大值
extern "C" void solve(const float* input, float* output, int N, int k) {
    if (N <= 0 || k <= 0) {
        return;
    }

    // 如果用户传入 k > N，就直接收缩成 N。
    // 否则输出前 k 大没有定义。
    if (k > N) {
        k = N;
    }

    // 把长度补到 2 的幂，满足 bitonic network 的前提。
    int padded_n = next_pow2(N);
    float* data = nullptr;
    cudaMalloc(&data, padded_n * sizeof(float));

    // 先拷贝，再 padding。
    int blocks = div_up(padded_n, kThreads);
    copy_and_pad_kernel<<<blocks, kThreads>>>(input, data, N, padded_n);

    // 标准 bitonic sort 双层循环：
    //
    // 外层 stage：
    //   决定当前在排多长的 bitonic 段，2 -> 4 -> 8 -> 16 ...
    //
    // 内层 stride：
    //   决定当前这一轮比较的距离，从大到小逐步收紧
    //
    // 这是最经典、最容易记忆的 bitonic launch 模板。
    for (int stage = 2; stage <= padded_n; stage <<= 1) {
        for (int stride = stage >> 1; stride > 0; stride >>= 1) {
            bitonic_step_kernel<<<blocks, kThreads>>>(data, stride, stage, padded_n);
        }
    }

    // 排完序后，最后 k 个元素就是 top-k 最大值。
    int out_blocks = div_up(k, kThreads);
    gather_topk_kernel<<<out_blocks, kThreads>>>(data, output, padded_n, k);

    cudaDeviceSynchronize();
    cudaFree(data);
}
