cmake_minimum_required(VERSION 3.17)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda-11.1/bin/nvcc)
project(cils_cuda LANGUAGES CUDA CXX)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(netCDF REQUIRED)
find_package(Eigen3 REQUIRED)

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

include_directories(${PYTHON_INCLUDE_DIRS})

link_libraries(${PYTHON_LIBRARIES})
link_libraries(Eigen3::Eigen)

add_executable(cils_cuda ils_search_cuda.cu)
target_link_libraries(cils_cuda Eigen3::Eigen)
target_link_libraries(cils_cuda OpenMP::OpenMP_CXX)
set_target_properties(cils_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


if (MPI_FOUND)
    target_link_libraries(cils_cuda ${MPI_LIBRARIES})
    message("-- MPI_FOUND")
endif ()

if (OpenMP_CXX_FOUND)
    target_link_libraries(cils_cuda OpenMP::OpenMP_CXX)
    message("-- OpenMP_CXX_FOUND")
endif ()

if (netCDF_FOUND)
    target_link_libraries(cils_cuda ${netCDF_LIBRARIES})
    message("-- netCDF_FOUND")
endif ()

if (Eigen3_FOUND)
    target_link_libraries(cils_cuda Eigen3::Eigen)
    message("-- Eigen3_FOUND")
endif ()
