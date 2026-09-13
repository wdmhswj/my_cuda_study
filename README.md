# my_cuda_study

## 参考资料
- 简单入门教程：https://developer.nvidia.cn/blog/even-easier-introduction-cuda-2/
- nvidia cuda 官方文档：https://docs.nvidia.com/cuda/
- NVIDIA curated collection of educational resources related to general purpose GPU programming: https://github.com/NVIDIA/accelerated-computing-hub
- [CUDA Platform for Accelerated Computing | NVIDIA Developer](https://developer.nvidia.com/cuda?ncid=GTC-NV0YIGWW&utm_source=chatgpt.com)
- [CUDA 入门教程：更简单的介绍 (更新版) - NVIDIA 技术博客](https://developer.nvidia.cn/blog/even-easier-introduction-cuda-2/)
- [PaddleJitLab/CUDATutorial: A self-learning tutorail for CUDA High Performance Programing.](https://github.com/PaddleJitLab/CUDATutorial)
- [alternbits/awesome-cuda-books: A curated list of best cuda programming books](https://github.com/alternbits/awesome-cuda-books)
- [harleyszhang/llm_note: LLM notes, including model inference, transformer model structure, and llm framework code analysis notes.](https://github.com/harleyszhang/llm_note)
- [xlite-dev/LeetCUDA: Modern CUDA Learn Notes with PyTorch for Beginners, 200+ CUDA Kernels, Tensor Cores, HGEMM, FA-2 MMA.](https://github.com/xlite-dev/LeetCUDA)


## Cuda Programming Guide

### Introduction to CUDA

#### Introduction
GPU devotes more transistors to data processing: ![GPU devotes more transistors to data processing](./assets/gpu-devotes-more-transistors-to-data-processing.png)

nvidia 提供了 cuBLAS, cuFFT, cuDNN, and CUTLASS 等库，用于实现各个领域的算法，同时这些库还为每种GPU架构进行了优化，因此开发者可以直接使用这些库来实现高性能的计算任务。

domain-specific语言，如：Nvidia的Warp和Openai的Triton，可以让直接在Cuda上编写高性能代码变得更加容易。


#### Programming Model

##### 🧠 一、CUDA 的整体思想：异构计算模型（Heterogeneous Model）

CUDA 假设系统由 **CPU + GPU** 共同组成：

- **Host（主机）**  
  - CPU 及其直接连接的内存  
  - 负责程序启动、内存管理、kernel 调度

- **Device（设备）**  
  - GPU 及其显存  
  - 负责大规模并行计算

关键认知：
- 程序 **始终从 CPU 开始执行**
- GPU 只执行被显式启动的 **kernel**
- CPU 与 GPU **可以并行工作**，性能优化的目标是让两者都“忙起来”

---

##### 🚀 二、Kernel 与线程层级结构（最核心）

###### 1️⃣ Kernel 是什么？

- Kernel 是 **在 GPU 上执行的函数**
- 一次 kernel 启动 ≈ 启动 **成千上万个线程**
- 每个线程执行 **同一段代码，但处理不同数据**

---

###### 2️⃣ 线程组织结构（非常重要）

CUDA 使用 **三级层次结构**：

| 层级 | 说明 |
|----|----|
| **Thread** | 最小执行单元 |
| **Thread Block** | 一组线程，在同一个 SM 上执行 |
| **Grid** | 所有 thread block 的集合 |

关键规则：
- **一个 thread block 只能在一个 SM 上执行**
- **不同 block 之间不能同步或通信**
- block 的执行顺序 **不确定**

👉 这直接决定了 CUDA 程序必须是 **block 级别无依赖的**

---

###### 3️⃣ 为什么要这样设计？

- GPU 的 SM 数量有限
- Grid 可以非常大（百万级 block）
- CUDA 通过 **动态调度 block** 实现可扩展性

> 这就是 CUDA 能在“1 个 SM 或 1000 个 SM”上都正确运行的根本原因

---

##### 🧩 三、Warp 与 SIMT 执行模型（性能关键）

###### 1️⃣ Warp 是什么？

- **32 个线程** 组成一个 warp
- Warp 是 **GPU 实际调度和执行的最小单位**

---

###### 2️⃣ SIMT（Single Instruction, Multiple Threads）

- 同一个 warp 内：
  - 执行 **同一条指令**
  - 每个线程有自己的寄存器和数据

⚠️ **Warp Divergence（分支发散）**

- 如果 warp 内线程走不同分支：
  - GPU 会“分批执行”
  - 未执行的线程被 mask 掉
- 结果：**性能下降**

👉 性能经验法则：
- 尽量让 warp 内线程走 **相同控制流**
- block 线程数最好是 **32 的倍数**

---

##### 🧱 四、GPU 硬件抽象模型

CUDA 把 GPU 抽象为：

- **GPU**
  - 由多个 **GPC（Graphics Processing Cluster）** 组成
- **GPC**
  - 包含多个 **SM（Streaming Multiprocessor）**
- **SM**
  - 执行 thread block
  - 拥有寄存器、共享内存、L1 cache

重要原则：
- CUDA 编程模型 **屏蔽具体硬件差异**
- 不同架构性能不同，但 **程序语义一致**

---

##### 🧠 五、Thread Block Cluster（新特性，Compute Capability ≥ 9.0）

这是较新的概念，重点理解用途即可：

- Cluster = 一组相邻的 thread block
- 所有 block：
  - **运行在同一个 GPC**
  - 可以跨 block 同步与通信
- 使用 **Cooperative Groups**

适用场景：
- 需要 **block 间协作**
- 但仍希望保持 CUDA 的调度模型

---

##### 💾 六、CUDA 内存模型（理解性能的关键）

###### 1️⃣ 全局内存（Global Memory）

- GPU 显存
- 所有 SM 可访问
- 延迟高、带宽大

---

###### 2️⃣ 片上内存（On-chip Memory）

| 类型 | 特点 |
|----|----|
| **Register** | 每线程私有，最快 |
| **Shared Memory** | block / cluster 共享 |
| **L1 Cache** | SM 内 |
| **L2 Cache** | GPU 全局共享 |

关键限制：
- 寄存器 & shared memory **容量有限**
- 资源使用过多 → kernel 无法 launch 或 occupancy 降低

---

###### 3️⃣ Unified Memory（统一内存）

- CPU / GPU 都能访问
- 由 CUDA runtime 自动迁移
- **方便但不等于高性能**

性能原则：
- 尽量让数据 **靠近使用它的处理器**
- 减少迁移次数

---

##### 🧭 七、CUDA 编程模型的“铁律”

这是整章最重要的总结：

1. **Block 之间不能有依赖**
2. **Block 执行顺序不可预测**
3. **Warp 是性能的基本单位**
4. **内存层级决定性能上限**
5. **模型保证正确性，不保证性能**

---

##### 🎯 给你一个“工程师级”的理解方式

可以这样记住 CUDA：

> CUDA 不是“写并行代码”，  
> 而是 **描述一个可以被无限拆分、任意调度的计算问题**

### Programming GPUs in CUDA
#### Intro to CUDA C++

- CUDA Runtime API: 是在C++中使用CUDA最常用的方式，它是建立在较为底层的CUDA Driver API之上的。
- nvcc: 在Nvidia Cuda中是使用NVIDIA CUDA Compiler: `nvcc`来编译以C++编写的GPU代码的。
- kernel: 是在GPU上执行并且从主机代码中调用的函数，kernel被设计为由多个并行线程同时执行

##### Kernels

kernel函数使用`__global__`修饰符进行声明，表示该函数将在GPU上执行，并且可以从主机代码中调用。kernel函数的返回类型必须是void。
```cpp
// Kernel definition
__global__ void vecAdd(float* A, float* B, float* C) {
  ... // Kernel code
}
```

执行kernels时并行线程的数量是作为kernel launch配置的一部分进行指定的。有2种kernel launch配置方式：triple chevron语法和`cudaLaunchKernel`函数。

##### Triple Chevron Notation
使用`<<<...>>>`语法来指定kernel的执行配置。第一个参数是线程块的数量，第二个参数是每个线程块中的线程数量。
```cpp
__global__ void vecAdd(float* A, float* B, float* C) {

}

int main() {
  ...
  // Kernel notation
  vecAdd<<<1, 256>>>(A, B, C);  // Launch kernel with 1 block of 256 threads
}
```
当使用2或3维的grid或block时，可以使用`dim3`类型来指定每个维度的大小。如下示例使用使用了16x16的线程块grid，且每个线程块包含8x8个线程。
```cpp
int main() {
  ...
  dim3 grid(16, 16);      // 16x16 blocks
  dim3 block(8, 8);      // 8x8 threads per block
  MatAdd<<<grid, block>>>(A, B, C);
  ...
}

##### Thread and Grid Index Intrinsics 线程和网格索引内置函数
CUDA提供了一些内置变量，用于获取当前线程和线程块的索引信息。这些变量包括：
- `threadIdx`: 当前线程在其线程块中的索引。
- `blockDim`: 线程块的维度。
- `blockIdx`: 当前线程块在网格中的索引。
- `gridDim`: 网格的维度。
这些内置变量都是`dim3`类型，可以通过它们的`.x`、`.y`和`.z`成员访问各个维度的索引。

```cpp
__global__ void vecAdd(float* A, float* B, float* C)
{
   // calculate which element this thread is responsible for computing
   int workIndex = threadIdx.x + blockDim.x * blockIdx.x

   // Perform computation
   C[workIndex] = A[workIndex] + B[workIndex];
}

int main()
{
    ...
    // A, B, and C are vectors of 1024 elements
    vecAdd<<<4, 256>>>(A, B, C);
    ...
}
```
以上代码中，`vecAdd` kernel被配置为使用4个线程块，每个线程块包含256个线程。每个线程计算一个向量元素的和。`workIndex`变量计算出每个线程负责处理的向量元素的索引。

##### Bounds Checking 边界检查
在kernel中，通常需要进行边界检查，以确保线程不会访问超出数组范围的内存。以下是一个示例：
```cpp
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
     // calculate which element this thread is responsible for computing
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x

     if(workIndex < vectorLength)
     {
         // Perform computation
         C[workIndex] = A[workIndex] + B[workIndex];
     }
}
```
在这个示例中，`vecAdd` kernel接受一个额外的参数`vectorLength`，用于指定向量的长度。每个线程在执行计算之前都会检查其`workIndex`是否小于`vectorLength`，以避免访问越界。

此外，所需线程块的数量可以通过以下公式计算：
```numBlocks = (vectorLength + threadsPerBlock - 1) / threadsPerBlock;
```
这个公式确保即使`vectorLength`不是`threadsPerBlock`的整数倍，也能覆盖所有元素。
```cpp
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = (vectorLength + threads-1)/threads;
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```
以上代码计算出所需的线程块数量，并启动`vecAdd` kernel。

```cpp
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = cuda::ceil_div(vectorLength, threads);
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```
以上示例则使用CUDA Core Compute Library(CCCL)中的`cuda::ceil_div`函数来计算所需的线程块数量。这个函数可以确保在进行整数除法时向上取整，从而避免了手动计算的复杂性。

##### Unified Memory 统一内存
统一内存是CUDA的一项特性，允许CPU和GPU共享同一块内存空间。使用统一内存可以简化内存管理，因为NVIDIA DRIVER保证了统一内存可以被CPU和GPU访问。
```cpp
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

}
```
以上示例展示了如何使用统一内存来简化CUDA程序的内存管理。通过`cudaMallocManaged`函数分配的内存可以被CPU和GPU共享，无需显式的数据传输。kernel执行完成后，调用`cudaDeviceSynchronize`确保GPU计算完成，然后可以直接在CPU上访问结果进行验证。最后，使用`cudaFree`释放统一内存。

##### Explicit Memory Management 显式内存管理
```cpp
void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    
    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // start-allocate-and-copy
    // Allocate memory on the GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));

    // Copy data to the GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));
    // end-allocate-and-copy

    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC);
    // wait for kernel execution to complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}
```
以上示例展示了如何使用显式内存管理来进行CUDA编程。首先，通过`cudaMallocHost`分配主机内存 (`cudaMallocHost`对于需要在CPU和GPU之间进行数据传输的缓冲区是最佳实践)。然后，使用`cudaMalloc`在GPU上分配设备内存，并通过`cudaMemcpy`将数据从主机复制到设备。kernel执行完成后，再次使用`cudaMemcpy`将结果从设备复制回主机。最后，释放所有分配的内存以避免内存泄漏。`cudaMemcpy`的最后一个参数的类型是`cudaMemcpyKind_t`枚举类型，它指定了数据传输的方向。在这个示例中，使用了`cudaMemcpyDefault`，它允许CUDA运行时根据指针的来源自动确定传输方向。

##### 完整实例代码
Unified Memory:
```cpp
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

//unified-memory-begin
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

}
//unified-memory-end


int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    unifiedMemExample(vectorLength);		
    return 0;
}
```

Explicit Memory Management:
```cpp
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

//explicit-memory-begin
void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    
    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // start-allocate-and-copy
    // Allocate memory on the GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));

    // Copy data to the GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));
    // end-allocate-and-copy

    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC);
    // wait for kernel execution to complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}
//explicit-memory-end


int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    explicitMemExample(vectorLength);		
    return 0;
}
```
以上2个示例展示了如何使用统一内存和显式内存管理来进行CUDA编程，可以通过以下命令编译和运行这些示例：
```bash
$ nvcc vecAdd_unifiedMemory.cu -o vecAdd_unifiedMemory
$ ./vecAdd_unifiedMemory
Unified Memory: CPU and GPU answers match
$ ./vecAdd_unifiedMemory 4096
Unified Memory: CPU and GPU answers match
```
```bash
$ nvcc vecAdd_explicitMemory.cu -o vecAdd_explicitMemory
$ ./vecAdd_explicitMemory
Explicit Memory: CPU and GPU answers match
$ ./vecAdd_explicitMemory 4096
Explicit Memory: CPU and GPU answers match
```
在这2个示例中，所有线程都独立并行地执行向量加法操作，并且不需要线程间的同步或通信。

在block层面的最基本的同步机制是`__syncthreads()`函数。这个函数用于在同一个block内的所有线程之间进行同步，确保所有线程在继续执行之前都达到了同步点。

##### Runtime Initialization 运行时初始化
CUDA运行时为系统中的每个设备创建一个CUDA上下文。当应用程序第一次调用CUDA运行时API时，CUDA运行时会自动初始化上下文。上下文初始化包括以下步骤：
1. 选择要使用的GPU设备（如果系统中有多个设备）。
2. 分配必要的资源（如内存、流等）。
3. 设置设备属性和状态。
开发者通常不需要显式地初始化CUDA上下文，因为CUDA运行时会在第一次调用API时自动完成这些步骤。然而，了解这一过程对于调试和优化CUDA应用程序是有帮助的。

##### Error Checking in CUDA
每个CUDA API调用都会返回一个枚举类型`cudaError_t`的值。当没有错误发生时，返回值为`cudaSuccess`。很多CUDA程序会实现一个宏来简化错误检查过程。例如：
```cpp
#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)
```
这个宏接受一个CUDA API调用作为参数，执行该调用并检查返回值。如果返回值不是`cudaSuccess`，则打印错误信息，包括文件名、行号和错误描述。do ...while(0)的结构确保宏在任何上下文中都能正确使用。这个宏使用了`cudaGetErrorString`API，返回一个描述错误代码的字符串。应用可以以使用这个宏来包装所有的CUDA API调用，以确保及时捕获和报告错误。例如：
```cpp
    CUDA_CHECK(cudaMalloc(&devA, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devC, vectorLength*sizeof(float)));
```

##### Error State
CUDA运行时为每个主机线程维护一个cudaError_t状态变量。其默认值为cudaSuccess，并在发生错误时被覆盖。cudaGetLastError函数会返回当前错误状态，然后将其重置为cudaSuccess。而cudaPeekLastError则仅返回错误状态而不进行重置。

使用三重尖括号（<<< >>>）启动的内核函数本身不返回cudaError_t。良好的实践是在内核启动后立即检查错误状态，以检测内核启动时的即时错误或内核启动前的异步错误。需要注意的是，在内核启动后立即检查得到的cudaSuccess返回值，并不代表内核已成功执行甚至开始执行；它仅验证了传递给运行时的内核启动参数和执行配置没有触发错误，且错误状态不是内核启动之前已有的旧错误或异步错误。

##### Asynchronous Errors 异步错误
CUDA 核函数的启动和许多运行时 API 都是异步的。异步的 CUDA 运行时 API 将在《异步执行》章节中详细讨论。CUDA 的错误状态会在每次发生错误时被设置和覆盖。这意味着，在异步操作执行期间发生的错误，只有当下一次检查错误状态时才会被报告。如前所述，这可能是调用 cudaGetLastError 或 cudaPeekLastError，也可能是任何返回 cudaError_t 的 CUDA API。 

当 CUDA 运行时 API 函数返回错误时，错误状态不会被自动清除。这意味着，由异步错误（例如内核的无效内存访问）产生的错误码，将会被每一个后续的 CUDA 运行时 API 返回，直到通过调用 cudaGetLastError 清除了错误状态为止。

```cpp
    vecAdd<<<blocks, threads>>>(devA, devB, devC);
    // check error state after kernel launch
    CUDA_CHECK(cudaGetLastError());
    // wait for kernel execution to complete
    // The CUDA_CHECK will report errors that occurred during execution of the kernel
    CUDA_CHECK(cudaDeviceSynchronize());
```
以上代码展示了如何在内核启动后立即检查错误状态，以捕获任何启动时的错误。随后，通过调用`cudaDeviceSynchronize`等待内核执行完成，并再次检查错误状态，以捕获在内核执行期间发生的任何异步错误。

##### CUDA_LOG_FILE
使用 CUDA_LOG_FILE 环境变量是识别 CUDA 错误的有效方法。设置该变量后，CUDA 驱动程序会将遇到的错误信息写入指定路径的文件中。例如，当内核启动的线程块尺寸超过架构支持的最大值时，常规错误检查仅报告“无效参数”，而 CUDA_LOG_FILE 生成的日志文件会提供更详细的错误描述，如具体超出限制的维度信息。
```cpp
__global__ void k()
{ }

int main()
{
        k<<<8192, 4096>>>(); // Invalid block size
        CUDA_CHECK(cudaGetLastError());
        return 0;
}
```

CUDA_LOG_FILE 可设置为 stdout 或 stderr，从而将错误直接输出到标准输出或标准错误流。这一方法即使在没有完善错误检查的应用程序中也能捕获 CUDA 错误，强化调试能力，但仅靠环境变量无法实现运行时的错误恢复。
```cpp
$ env CUDA_LOG_FILE=cudaLog.txt ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
$ cat cudaLog.txt
[12:46:23.854][137216133754880][CUDA][E] One or more of block dimensions of (4096,1,1) exceeds corresponding maximum value of (1024,1024,64)
[12:46:23.854][137216133754880][CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuLaunchKernel
```

此外，CUDA 的错误日志管理功能允许注册回调函数，在检测到错误时自动调用，便于运行时捕获、处理错误，并与应用程序现有日志系统集成。该功能需要 NVIDIA 驱动程序版本 r570 或更高版本支持。

##### Device and Host Functions 设备和主机函数
__global__ 指定符用于标识内核的入口点，即在 GPU 上并行执行的函数。内核通常由主机端启动，但也可通过动态并行从其他内核内部启动。

__device__ 指定符表示函数应编译为在 GPU 上执行，并可由其他 __device__ 或 __global__ 函数调用。函数（包括类成员函数、仿函数和 Lambda 表达式）可同时被指定为 __device__ 和 __host__。

##### Variable Specifiers 变量指定符
CUDA 存储说明符用于控制静态变量的内存位置：
- __device__ 表示变量存储在全局内存中；
- __constant__ 表示变量存储在常量内存中；
- __managed__ 表示变量以统一内存形式存储；
- __shared__ 表示变量存储在共享内存中。

若在 __device__ 或 __global__ 函数内声明变量时未使用说明符，变量会尽可能分配至寄存器，必要时则存入本地内存。而在 __device__ 或 __global__ 函数外部未使用说明符声明的任何变量，都将分配在系统内存中。

##### Detecting Device Compilation
当函数使用 __host__ 或 __device__ 修饰时，编译器会同时生成GPU和CPU代码。为了在函数中指定仅适用于GPU或CPU的代码，可以使用预处理器检查 CUDA_ARCH 是否定义，这是最常用的方法。

##### Launching with Clusters in Triple Chevron Notation
线程块集群可通过两种方式启用：一是在内核编译时使用__cluster_dims__(X,Y,Z)属性，二是通过CUDA内核启动API cudaLaunchKernelEx。使用编译时属性时，集群大小在编译阶段即固定，之后可采用传统的<<< , >>>语法启动内核。一旦采用编译时设置的集群大小，启动内核时便无法再修改该集群尺寸。

```cpp
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```
在上述示例中，`cluster_kernel`内核函数被编译为使用2个线程块集群（X维度）和1个线程块集群（Y和Z维度）。在内核启动时，仍然使用传统的<<< , >>>语法指定线程块数量和每个线程块中的线程数量。需要注意的是，网格维度必须是集群大小的整数倍。


