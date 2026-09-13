enable_language(CUDA)

add_executable(01_build_dev_env
    01_build_dev_env/hello_world.cu
)

add_executable(vector_add_cpu
    02_first_kernel/vector_add_cpu.cu
)
add_executable(vector_add_gpu
    02_first_kernel/vector_add_gpu.cu
)
add_executable(vector_add_gpu_1
    02_first_kernel/vector_add_gpu_1.cu
)

add_executable(vector_add_thread
    04_first_refine_kernel/vector_add_thread.cu
)
add_executable(vector_add_grid
    04_first_refine_kernel/vector_add_grid.cu
)

add_executable(matmmul_raw
    06_impl_matmul/matmul_raw.cu
)