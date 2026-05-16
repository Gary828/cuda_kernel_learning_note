#include <cuda_runtime.h>
#include <math.h>

__global__ void headwise_QKT(const float* Q, const float* K, float* output, int N, int d_model, int h) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int row  = blockIdx.y * blockDim.y + threadIdx.y;
    int col  = blockIdx.z * blockDim.z + threadIdx.z;

    if (head < h && row < N && col < N) {
        int dk = d_model / h;
        float val = 0.0f;
        for (int i = 0; i < dk; i++) {
            float q = Q[row * d_model + head * dk + i];
            float k = K[col * d_model + head * dk + i];  // row-major access
            val += q * k;
        }
        output[head * N * N + row * N + col] = val / sqrtf((float)dk);
    }
}

__global__ void headwise_softmax(const float* qkt, float* output, int N, int h) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int row  = blockIdx.y * blockDim.y + threadIdx.y;

    if (head < h && row < N) {
        float max_val = -INFINITY;
        for (int col = 0; col < N; col++) {
            max_val = fmaxf(max_val, qkt[head * N * N + row * N + col]);
        }

        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            float val = expf(qkt[head * N * N + row * N + col] - max_val);
            output[head * N * N + row * N + col] = val;
            sum += val;
        }

        for (int col = 0; col < N; col++) {
            output[head * N * N + row * N + col] /= sum;
        }
    }
}

__global__ void headwise_SV(const float* S, const float* V, float* output, int N, int d_model, int h) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int row  = blockIdx.y * blockDim.y + threadIdx.y;
    int col  = blockIdx.z * blockDim.z + threadIdx.z;

    int dk = d_model / h;
    if (head < h && row < N && col < dk) {
        float val = 0.0f;
        for (int i = 0; i < N; i++) {
            float s = S[head * N * N + row * N + i];
            float v = V[i * d_model + head * dk + col];
            val += s * v;
        }
        output[row * d_model + head * dk + col] = val;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
    int dk = d_model / h;
    float *qkt, *sftmax;

    cudaMalloc(&qkt, h * N * N * sizeof(float));
    cudaMalloc(&sftmax, h * N * N * sizeof(float));
    cudaMemset(output, 0, N * d_model * sizeof(float));

    dim3 threads1(8, 8, 8);
    dim3 blocks1((h + threads1.x - 1) / threads1.x,
                 (N + threads1.y - 1) / threads1.y,
                 (N + threads1.z - 1) / threads1.z);
    headwise_QKT<<<blocks1, threads1>>>(Q, K, qkt, N, d_model, h);
    cudaDeviceSynchronize();

    dim3 threads2(8, 8);
    dim3 blocks2((h + threads2.x - 1) / threads2.x,
                 (N + threads2.y - 1) / threads2.y);
    headwise_softmax<<<blocks2, threads2>>>(qkt, sftmax, N, h);
    cudaDeviceSynchronize();

    dim3 threads3(8, 8, 8);
    dim3 blocks3((h + threads3.x - 1) / threads3.x,
                 (N + threads3.y - 1) / threads3.y,
                 (dk + threads3.z - 1) / threads3.z);
    headwise_SV<<<blocks3, threads3>>>(sftmax, V, output, N, d_model, h);
    cudaDeviceSynchronize();

    cudaFree(qkt);
    cudaFree(sftmax);
}
