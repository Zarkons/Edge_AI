cmake_minimum_required(VERSION 3.16)
project(my_custom_kernel_app CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point this to your existing tt-metal build folder.
set(TT_METAL_BUILD_DIR "/home/cesljarov.linux/tt-metal/build" CACHE PATH "Path to tt-metal build directory")
set(TT_METAL_CMAKE_DIR "${TT_METAL_BUILD_DIR}/lib/cmake/tt-metalium" CACHE PATH "Path to tt-metalium CMake package directory")

# Use the exported tt-metalium package from the build output tree.
find_package(TT-Metalium CONFIG REQUIRED PATHS "${TT_METAL_CMAKE_DIR}" NO_DEFAULT_PATH)

add_executable(my_app main.cpp)

# Pull in tt-metal include dirs and transitive dependencies (nlohmann_json, fmt, umd, etc.).
target_link_libraries(my_app PRIVATE TT::Metalium)
