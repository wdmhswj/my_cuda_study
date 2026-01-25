# my_cuda_study

## 参考资料
- 简单入门教程：https://developer.nvidia.cn/blog/even-easier-introduction-cuda-2/
- nvidia cuda 官方文档：https://docs.nvidia.com/cuda/
- NVIDIA curated collection of educational resources related to general purpose GPU programming: https://github.com/NVIDIA/accelerated-computing-hub

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