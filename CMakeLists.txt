cmake_minimum_required(VERSION 3.10)
project(knn_app)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include header files
include_directories(include)

# Define the source files
add_executable(knn_app src/main.cpp)
