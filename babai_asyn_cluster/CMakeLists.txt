cmake_minimum_required(VERSION 3.8.2)
project(babai_asyn)

#FIND_PACKAGE(PythonInterp 3)
#find_package(PythonLibs 3.8 REQUIRED)
#message("Python_FOUND:${Python_FOUND}")
#find_package(PythonLibs 3.8 COMPONENTS Interpreter NumPy)
find_package(OpenMP REQUIRED)
#find_package(MPI REQUIRED)
#find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_STANDARD 17)

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

#include_directories(${PYTHON_INCLUDE_DIRS})
#include_directories(${MPI_INCLUDE_PATH})
include_directories("${EIGEN3_INCLUDE_DIR}")

link_libraries(${PYTHON_LIBRARIES})

add_executable(babai_asyn Babai_search_asyn.h Babai_search_asyn.cpp)#matplotlibcpp.h)
#target_link_libraries(babai_asyn Eigen3::Eigen)
target_link_libraries(babai_asyn OpenMP::OpenMP_CXX)
#target_link_libraries(babai_asyn ${MPI_C_LIBRARIES})