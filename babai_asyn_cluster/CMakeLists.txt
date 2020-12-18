cmake_minimum_required(VERSION 3.8.2)
project(cils)

# detect C++ compiler - the C compiler must be the same type too
# to compile with OpenMP
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopenmp")
    message(STATUS "Compiling with OpenMP for Intel")
else()
    message(SEND_ERROR "Don't know OpenMP flag for this compiler!")
endif()
message(HERE!!!)

find_package(OpenMP REQUIRED)
#find_package(OpenCL REQUIRED)
find_package(MPI REQUIRED)
find_package(netCDF REQUIRED)
#find_package(Eigen3 REQUIRED)
set(CMAKE_CXX_STANDARD 11)

if (MPI_FOUND)
    include_directories(${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
    if(SLURM)
        set(MPIEXEC "srun")
        set(MPIOPTS "--cpu-bind=cores")
    else()
        set(MPIEXEC "mpirun")
        set(MPIOPTS "")
    endif(SLURM)

endif ()

add_executable(cils main.cpp)

if (MPI_FOUND)
    target_link_libraries(cils ${MPI_LIBRARIES})
    message("-- MPI_FOUND")
endif ()

#if (OpenMP_CXX_FOUND)
#    target_link_libraries(cils OpenMP::OpenMP_CXX)
#    message("-- OpenMP_CXX_FOUND")
#endif ()

if (netCDF_FOUND)
    target_link_libraries(cils ${netCDF_LIBRARIES})
    message("-- netCDF_FOUND")
endif ()

if (Eigen3_FOUND)
    target_link_libraries(cils Eigen3::Eigen)
    message("-- Eigen3_FOUND")
endif ()

#if (OpenCL_FOUND)
#    target_link_libraries(cils ${OPENCL_LIBRARIES})
#    message("-- OpenCL_FOUND")
#endif ()