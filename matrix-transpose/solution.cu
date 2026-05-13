#include <cuda_runtime.h>

#define TILE_SIZE 16
__global__ void matrix_transpose_kernel(const float* input, float* output, int M, int N){
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float sdata[TILE_SIZE][TILE_SIZE + 1]; // padding

    int x = bx * blcokDim.x + tx;
    int y = by * blockDim.y + ty;
    
    if(y < M && x < N)
        sdata[ty][tx] = input[y * N + x];
    __syncthreads();

    x = by * blockDim.y + tx;
    y = bx * blockDim.x + ty;
    if(y < N && x < M)
        output[y * M + x] = sdata[tx][ty];
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((cols + TILE_SIZE - 1) / TILE_SIZE,
                       (rows + TILE_SIZE - 1) / TILE_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}