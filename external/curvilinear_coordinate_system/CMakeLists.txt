project(crccosy)
cmake_minimum_required(VERSION 3.0)

option(ADD_TESTS "unit tests" OFF)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -Wno-ignored-attributes)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

add_subdirectory(cpp)

if (ADD_PYTHON_BINDINGS)
    set(PYTHON_INCLUDE_DIR ${PATH_TO_PYTHON_ENVIRONMENT}/include/python${PYTHON_VERSION}m)
    set(PYTHON_LIBRARY ${PATH_TO_PYTHON_ENVIRONMENT}/lib/libpython${PYTHON_VERSION}m.so)
    set(PYTHON_EXECUTABLE ${PATH_TO_PYTHON_ENVIRONMENT}/bin/python${PYTHON_VERSION}m)
    add_subdirectory(external/pybind11)
    add_subdirectory(python_binding)
endif ()



