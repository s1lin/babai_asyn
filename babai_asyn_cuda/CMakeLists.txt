cmake_minimum_required(VERSION 3.17)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda-11.1/bin/nvcc)
project(babai_asyn_cuda LANGUAGES CUDA CXX)

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()


add_executable(babai_asyn_cuda Babai_search_asyn.cu ../babai_asyn_c++/Babai_search_asyn.h)
target_link_libraries(babai_asyn_cuda Eigen3::Eigen)
target_link_libraries(babai_asyn_cuda OpenMP::OpenMP_CXX)
set_target_properties(babai_asyn_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)