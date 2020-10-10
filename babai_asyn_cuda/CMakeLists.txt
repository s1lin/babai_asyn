cmake_minimum_required(VERSION 3.17)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda-11.1/bin/nvcc)
project(babai_asyn_cuda CUDA)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
set(CMAKE_CUDA_STANDARD 11)

add_executable(babai_asyn_cuda Babai_search_asyn.cu Babai_search_asyn.cuh)
target_link_libraries (babai_asyn_cuda Eigen3::Eigen)
set_target_properties(
        babai_asyn_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)