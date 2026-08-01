#!/bin/bash

echo "🔨 Attempting compilation..."

TT_METAL_BUILD_DIR="/home/cesljarov.linux/tt-metal/build"
TT_METAL_CMAKE_DIR="${TT_METAL_BUILD_DIR}/lib/cmake/tt-metalium"

# 1. Optional clean step based on script arguments
if [ "$1" == "clean" ] || [ "$1" == "--clean" ]; then
    echo "🧹 Clean argument detected! Wiping previous builds and JIT kernel cache..."
    rm -rf build
    rm -rf ~/.tt_metal_built_kernels/
fi

# 2. Enter the build directory
mkdir -p build
cd build

# 3. Configure the project with CMake
echo "⚙️  Configuring with CMake..."
cmake .. \
    -DTT_METAL_BUILD_DIR="${TT_METAL_BUILD_DIR}" \
    -DTT_METAL_CMAKE_DIR="${TT_METAL_CMAKE_DIR}" \
    -DCMAKE_PREFIX_PATH="${TT_METAL_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Debug
if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed!"
    cd ..
    return 1 2>/dev/null || exit 1
fi

# 4. Compile the binary
echo "🏗️  Compiling application..."
cmake --build . -j4
if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    cd ..
    return 1 2>/dev/null || exit 1
fi

# 5. Return to the root folder cleanly
cd ..

echo "=================================================="
echo "🎉 Build finished successfully!"
echo "👉 Run your app using: ./build/my_app"
echo "=================================================="
