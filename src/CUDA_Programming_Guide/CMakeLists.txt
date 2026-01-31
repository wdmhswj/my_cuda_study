enable_language(CUDA)

add_executable(vecAdd_unifiedMemory
    vecAdd_unifiedMemory.cu
)


add_executable(vecAdd_explicitMemory
    vecAdd_explicitMemory.cu
)