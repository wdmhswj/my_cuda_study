cmake_minimum_required(VERSION 3.20)

project(my_cuda_study LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 统一输出目录（可选）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


add_subdirectory(src/An_Easier_Introduction_to_Cuda)
add_subdirectory(src/CUDA_Programming_Guide)
add_subdirectory(src/CUDATutorial)


# # 常用警告（可选）
# target_compile_features(add PRIVATE cxx_std_17)
# if (MSVC)
#     target_compile_options(add PRIVATE /W4)
# else()
#     target_compile_options(add PRIVATE -Wall -Wextra -Wpedantic)
# endif()