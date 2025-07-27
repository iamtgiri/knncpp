cmake_minimum_required(VERSION 3.10)
project(knn)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
add_executable(knn
    src/main.cpp
    
)
