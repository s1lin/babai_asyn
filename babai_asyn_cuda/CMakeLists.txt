cmake_minimum_required(VERSION 3.17)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda-11.1/bin/nvcc)
project(babai_asyn_cuda LANGUAGES CUDA CXX)

FIND_PACKAGE(PythonInterp 3)
find_package(PythonLibs 3.8 REQUIRED)
message("Python_FOUND:${Python_FOUND}")
find_package(PythonLibs 3.8 COMPONENTS Interpreter NumPy)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

include_directories(${PYTHON_INCLUDE_DIRS})

link_libraries(${PYTHON_LIBRARIES})
link_libraries(Eigen3::Eigen)

add_executable(babai_asyn_cuda Babai_search_asyn.cu ../babai_asyn_c++/src/include/sils.h)
target_link_libraries(babai_asyn_cuda Eigen3::Eigen)
target_link_libraries(babai_asyn_cuda OpenMP::OpenMP_CXX)
set_target_properties(babai_asyn_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)