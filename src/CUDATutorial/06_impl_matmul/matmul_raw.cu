#include <stdio.h>
#include <cuda_runtime.h>

void randomize_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = rand() % 100;
    }
}

void sgemm_naive_cpu(float *A, float *B, float *C, int M, int N, int K) {
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void sgemm_naive_kernel(float *A, float *B, float *C, int M, int N, int K) {
    uint row = blockIdx.x * blockDim.x + threadIdx.x;
    uint col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))


int main() {
    int m = 256;
    int n = 256 + 256;
    int k = 256;

    // 矩阵指针
    float *A, *B, *C, *C_ref; // host pointers
    float* d_A, *d_B, *d_C; // device pointers

    A = new float[m * k];
    B = new float[k * n];
    C = new float[m * n];
    C_ref = new float[m * n];

    randomize_matrix(A, m * k);
    randomize_matrix(B, k * n);

    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    sgemm_naive_cpu(A, B, C_ref, m, n, k);
    
    dim3 block_size(32, 32); // 每个 block 32x32 个线程，未指定的维度默认为 1
    dim3 grid_size(CEIL_DIV(m, 32), CEIL_DIV(n, 32)); // 计算需要多少个 block 来覆盖整个矩阵
    sgemm_naive_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < m * n; i++) {
        if (C[i] != C_ref[i]) {
            printf("Mismatch at index %d: GPU result = %f, CPU result = %f\n", i, C[i], C_ref[i]);
            return -1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_ref;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    return 0;
}