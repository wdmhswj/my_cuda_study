#include <stdio.h>

__global__ void add_kernel(float *x, float *y, float *out, int n){
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start_idx; i < n; i += stride){
        out[i] = x[i] + y[i];
    }
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

    float *cuda_x, *cuda_y, *cuda_out;
    cudaMalloc(&cuda_x, mem_size);
    cudaMalloc(&cuda_y, mem_size);
    cudaMalloc(&cuda_out, mem_size);
    cudaMemcpy(cuda_x, x, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y, y, mem_size, cudaMemcpyHostToDevice);

    add_kernel<<<1,256>>>(cuda_x, cuda_y, cuda_out, N);

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