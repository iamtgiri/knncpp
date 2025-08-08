# # # CMakeLists.txt

cmake_minimum_required(VERSION 3.10) # mandatory
project(knn_kdtree)                 # mandatory

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags (optional but recommended)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# OpenMP support
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include) # mandatory

# Source files
set(SOURCES                                        # mandatory
    src/knn_cli_main.cpp
)

# Define executable
add_executable(knn_kdtree ${SOURCES})               # mandatory

# Link OpenMP
target_link_libraries(knn_kdtree PRIVATE OpenMP::OpenMP_CXX)

