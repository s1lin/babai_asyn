cmake_minimum_required(VERSION 3.8)
project(cils_solver)

# detect C++ compiler - the C compiler must be the same type too
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CXX_COMPILER_GNUCLANG TRUE)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        set(CXX_COMPILER_GNU TRUE)
        message(STATUS "GNU C++ compiler detected")
        if (NOT ("${CMAKE_C_COMPILER_ID}" MATCHES "GNU"))
            message(SEND_ERROR "C and C++ compilers should match!")
        endif ()
    else ()
        set(CXX_COMPILER_CLANG TRUE)
        message(STATUS "Clang C++ compiler detected")
        if (NOT ("${CMAKE_C_COMPILER_ID}" MATCHES "Clang"))
            message(SEND_ERROR "C and C++ compilers should match!")
        endif ()
    endif ()
else ()
    message(SEND_ERROR "Could not identify compiler!")
endif ()

# Basic options
if (CXX_COMPILER_GNUCLANG OR CXX_COMPILER_INTEL)
    set(CMAKE_CXX_FLAGS "-std=c++17")
    set(CMAKE_C_FLAGS "-std=c17")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif ()

# to compile with OpenMP
if (CXX_COMPILER_GNUCLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    message(STATUS "Compiling with OpenMP for GNU")
elseif (CXX_COMPILER_INTEL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopenmp")
    message(STATUS "Compiling with OpenMP for Intel")
elseif (CXX_COMPILER_CRAY)
    message(STATUS "Compiling with OpenMP for Cray")
elseif (CXX_COMPILER_PGI)
    message(STATUS "Compiling with OpenMP for PGI")
else ()
    message(SEND_ERROR "Don't know OpenMP flag for this compiler!")
endif ()


find_package(Python3 COMPONENTS Interpreter NumPy)
find_package(OpenMP REQUIRED)
find_package(Matlab REQUIRED)
find_package(Boost REQUIRED)
#find_package(MPI)
set(CMAKE_CXX_STANDARD 17)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIR})
include_directories(${Matlab_INCLUDE_DIRS})
add_executable(cils main.cpp)


if (Python3_NumPy_FOUND)
    message("-- Python3_NumPy_FOUND ${Python3_NumPy_INCLUDE_DIR}")
endif ()

if (Python3_FOUND)
    message("-- Python3_FOUND ${Python3_LIBRARIES}")
    message("-- Python3_Interpreter_FOUND ${Python3_Interpreter_FOUND}")
    message("-- Python3_EXECUTABLE ${Python3_EXECUTABLE}")
    message("-- Python3_SITELIB ${Python3_SITELIB}")
    message("-- Python3_INCLUDE_DIRS ${Python3_INCLUDE_DIRS}")
    target_link_libraries(cils ${Python3_LIBRARIES})

endif ()

if (Matlab_FOUND)
    message(STATUS "Linking Matlab")
    target_link_libraries(cils ${Matlab_ROOT_DIR}/extern/bin/glnxa64/libMatlabEngine.so)
    target_link_libraries(cils ${Matlab_ROOT_DIR}/extern/bin/glnxa64/libMatlabDataArray.so)
    message("-- Matlab_FOUND ${PROJECT_NAME} ${Matlab_LIBRARIES}")
endif ()

if (Boost_FOUND)
    message(STATUS "Linking Boost")
    target_link_libraries(cils ${Boost_LIBRARY_DIRS})
    message("-- Boost_FOUND ${Boost_LIBRARY_DIRS}")
endif ()


#message("-- reduction PRIVATE ${reduction_lib}")
#cd ~/babai_asyn/babai_asyn_c++/ && cd build  && git pull && cmake .. -D Python3_NumPy_INCLUDE_DIR=/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/scipy-stack/2020b/lib/python3.8/site-packages/numpy/core/include/ && make -j 4 && cp cils ~/scratch/cils && cd ~/scratch/ && sbatch job_script_LLL.sh