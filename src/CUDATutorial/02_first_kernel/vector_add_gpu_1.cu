#include <stdio.h>

__global__ void add_kernel(float *x, float *y, float *out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        out[idx] = x[idx] + y[idx];
    }
    // for (int i = 0; i < n; ++i){
    //     out[i] = x[i] + y[i];
    // }
}

int main(){
    int N = 10000000;
    size_t mem_size = sizeof(float) * N;

    float *x, *y, *out;
    x = static_cast<float*>(malloc(mem_size));
    y = static_cast<float*>(malloc(mem_size));
    out = static_cast<float*>(malloc(mem_size));

    for(int i = 0; i < N; ++i){
        x[i] = 1.0;
        y[i] = 2.0;
    }

    int threads_per_block = 256; // 每个block的线程数
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block; // 计算需要多少个block来处理N个元素

    float *cuda_x, *cuda_y, *cuda_out;
    cudaMalloc(&cuda_x, mem_size);
    cudaMalloc(&cuda_y, mem_size);
    cudaMalloc(&cuda_out, mem_size);
    cudaMemcpy(cuda_x, x, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y, y, mem_size, cudaMemcpyHostToDevice);

    add_kernel<<<blocks_per_grid, threads_per_block>>>(cuda_x, cuda_y, cuda_out, N);

    cudaMemcpy(out, cuda_out, mem_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();  // 同步

    for(int i = 0; i < 10; ++i){
        printf("out[%d] = %.3f\n", i, out[i]);
    }

    cudaFree(cuda_x);
    cudaFree(cuda_y);
    cudaFree(cuda_out);

    free(x);
    free(y);
    free(out);
}