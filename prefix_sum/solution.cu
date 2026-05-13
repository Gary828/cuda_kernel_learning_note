#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

// block 内 scan
__global__ void scan_blocks(const float* input, float* output, float* segment_sum, int N) {
    __shared__ float buffers[2][BLOCK_SIZE];
    int in = 0, out = 1;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int lane = threadIdx.x;

    // load
    float val = (tid < N) ? input[tid] : 0.0f;
    buffers[in][lane] = val;
    __syncthreads();

    // inclusive scan in block
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        float temp = buffers[in][lane];
        if (lane >= stride) temp += buffers[in][lane - stride];
        buffers[out][lane] = temp;
        __syncthreads();
        int tmp = in; in = out; out = tmp;
    }

    if (tid < N) {
        output[tid] = buffers[in][lane];
    }

    if (lane == blockDim.x - 1) {
        segment_sum[blockIdx.x] = buffers[in][lane];
    }
}

// 给 output 每个 block 加偏移
__global__ void add_offsets(float* output, const float* offsets, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N) {
        int block = blockIdx.x;
        if (block > 0) {
            output[tid] += offsets[block - 1];
        }
    }
}

// 递归 scan
void gpu_scan(const float* input, float* output, int N) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 如果只需要一个 block，直接做 scan
    if (blocksPerGrid == 1) {
        float* dummy;
        cudaMalloc(&dummy, sizeof(float));
        scan_blocks<<<1, threadsPerBlock>>>(input, output, dummy, N);
        cudaFree(dummy);
        return;
    }

    // 分配 segment sum
    float* d_segment_sum;
    cudaMalloc(&d_segment_sum, sizeof(float) * blocksPerGrid);

    // step 1: block 内 scan
    scan_blocks<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_segment_sum, N);

    // step 2: 对 segment_sum 再 scan (递归调用)
    float* d_segment_sum_scan;
    cudaMalloc(&d_segment_sum_scan, sizeof(float) * blocksPerGrid);
    gpu_scan(d_segment_sum, d_segment_sum_scan, blocksPerGrid);

    // step 3: 把偏移加回
    add_offsets<<<blocksPerGrid, threadsPerBlock>>>(output, d_segment_sum_scan, N);

    cudaFree(d_segment_sum);
    cudaFree(d_segment_sum_scan);
}

// 外部接口
extern "C" void solve(const float* input, float* output, int N) {
    gpu_scan(input, output, N);
    cudaDeviceSynchronize();
}