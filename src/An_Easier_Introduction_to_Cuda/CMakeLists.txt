
# 启用 CUDA（让 CMake 使用 nvcc 来编译 .cu）
enable_language(CUDA)

add_executable(add
    add.cpp
)

# 等价于：nvcc add.cu -o add_cuda
add_executable(add_cuda
    add.cu
)

add_executable(add_grid
    add_grid.cu
)

# # 可选：保证与顶层一致的标准/分离编译行为（按需）
# set_target_properties(add_cuda PROPERTIES
#     CUDA_STANDARD 17
#     CUDA_STANDARD_REQUIRED ON
# )
