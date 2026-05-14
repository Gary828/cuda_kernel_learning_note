  #include <cuda_runtime.h>
  #include <math_constants.h>

  static inline int next_pow2(int x) {
      int p = 1;
      while (p < x) p <<= 1;
      return p;
  }

  __global__ void copy_and_pad_kernel(const float* input, float* data, int N, int paddedN) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < paddedN) {
          data[idx] = (idx < N) ? input[idx] : -CUDART_INF_F;
      }
  }

  __global__ void bitonic_sort_step_kernel(float* data, int j, int k, int N) {
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= (unsigned)N) return;

      unsigned int ixj = i ^ j;
      if (ixj > i && ixj < (unsigned)N) {
          float a = data[i];
          float b = data[ixj];

          // 标准 bitonic：最终得到升序
          bool ascending = ((i & k) == 0);

          if ((ascending && a > b) || (!ascending && a < b)) {
              data[i] = b;
              data[ixj] = a;
          }
      }
  }

  __global__ void gather_topk_kernel(const float* data, float* output, int paddedN, int k) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < k) {
          // data 是升序，最后 k 个是最大的，倒序写出变成降序 top-k
          output[idx] = data[paddedN - 1 - idx];
      }
  }

  // input, output are device pointers
  extern "C" void solve(const float* input, float* output, int N, int k) {
      if (N <= 0 || k <= 0) return;
      if (k > N) k = N;

      int paddedN = next_pow2(N);

      float* data = nullptr;
      cudaMalloc(&data, paddedN * sizeof(float));

      const int threads = 256;
      int blocks = (paddedN + threads - 1) / threads;

      copy_and_pad_kernel<<<blocks, threads>>>(input, data, N, paddedN);

      // 全局 bitonic sort
      for (int size = 2; size <= paddedN; size <<= 1) {
          for (int stride = size >> 1; stride > 0; stride >>= 1) {
              bitonic_sort_step_kernel<<<blocks, threads>>>(data, stride, size, paddedN);
          }
      }

      int outBlocks = (k + threads - 1) / threads;
      gather_topk_kernel<<<outBlocks, threads>>>(data, output, paddedN, k);

      cudaDeviceSynchronize();
      cudaFree(data);
  }
