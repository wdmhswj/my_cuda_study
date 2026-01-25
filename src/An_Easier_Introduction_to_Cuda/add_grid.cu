#include <iostream>
#include <math.h>


// kernel 函数：在GPU上执行，将2个数组的元素相加
__global__ // 表示这是一个CUDA内核函数
void add(int n, float* x, float* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        y[i] += x[i];
    }
}

int main()
{
    int N = 1 << 20; // 1M elements
    float* x,* y;

    // 分配统一内存：CPU和GPU都可以访问
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // 初始化数组
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // 在GPU上运行kernel函数
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // 等待GPU完成
    cudaDeviceSynchronize();

    // check 错误
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // 释放内存
    cudaFree(x);
    cudaFree(y);

    return 0;
}