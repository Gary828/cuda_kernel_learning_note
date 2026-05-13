#include <cuda_runtime.h>
#include <math.h>
__global__ void silu_kernel(const float* input, float* output, int N){
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < N){
        float value = input[gid];
        output[gid] = value / (1 + __expf(-value));
    }
}
