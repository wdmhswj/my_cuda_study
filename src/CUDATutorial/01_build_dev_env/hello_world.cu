#include <stdio.h>

__global__ void cuda_say_hello(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello world, CUDA! %d\n", idx);
}

int main(){
    printf("Hello world, CPU\n");
    cuda_say_hello<<<3,5>>>();

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    return 0;
}