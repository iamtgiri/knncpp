# # # CMakeLists.txt

# ------------------------------------------------------------------
# # Minimum required CMake version
# cmake_minimum_required(VERSION 3.10)

# # Project name and supported language
# project(MyCppProject LANGUAGES CXX)

# # C++ standard
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Enable compiler optimizations
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# # Find and enable OpenMP
# find_package(OpenMP REQUIRED)
# if(OpenMP_CXX_FOUND)
#     message(STATUS "✅ OpenMP support enabled")
# endif()

# # Include directory (e.g., for your header files)
# include_directories(${PROJECT_SOURCE_DIR}/include)

# # Source files
# set(SOURCES
#     src/main.cpp
#     src/your_module.cpp
#     src/another_module.cpp
# )

# # Add executable
# add_executable(${PROJECT_NAME} ${SOURCES})

# # Link libraries
# target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

# # Optional: Set output directory
# # set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
# -------------------------------------------------------------------------



# cmake_minimum_required(VERSION 3.10)
# project(knn)

# set(CMAKE_CXX_STANDARD 17)

# # Enable compiler optimizations (optional but recommended for speed)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# # Find and enable OpenMP
# find_package(OpenMP REQUIRED)

# # Include directories
# include_directories(${PROJECT_SOURCE_DIR}/include)

# # Source files
# add_executable(knn
#     # src/main.cpp
#     # src/knn_tester_3.cpp        # if you have a .cpp for your custom Vector class
#     src/main_kdtree.cpp
# )

# # Link OpenMP to your executable
# target_link_libraries(knn PRIVATE OpenMP::OpenMP_CXX)

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

