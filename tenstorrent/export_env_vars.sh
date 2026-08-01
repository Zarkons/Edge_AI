#!/bin/bash

# ==============================================================================
# TENSTORRENT TTSIM & TT-METAL ENVIRONMENT SETUP SCRIPT
# ==============================================================================

# 1. Base Project Paths
export TT_METAL_HOME="/home/cesljarov.linux/tt-metal"
export TT_METAL_SIMULATOR_HOME="/home/cesljarov.linux/tenstorrent/ttsim"
export TT_METAL_RUNTIME_ROOT="/home/cesljarov.linux/tt-metal"

# 2. Shared Libraries Setup
# Links libtt_metal.so and its dependent helper libraries from the build folder
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH"

# 3. Simulator Binary Mappings
# Maps the specific local .so simulation framework library file
export TT_SIM_LIB_PATH="$TT_METAL_SIMULATOR_HOME/libttsim_wh_aarch64.so"
export TT_METAL_SIMULATOR="$TT_METAL_SIMULATOR_HOME/libttsim_wh_aarch64.so"

# 4. Hardware Simulation Targets
export ARCH_NAME="wormhole_b0"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DPRINT_CORES="all"

# ==============================================================================
# VERIFICATION CHECKS
# ==============================================================================
echo "=================================================="
echo "🍏 Tenstorrent Simulation Environment Configured"
echo "=================================================="
echo "Target Architecture : $ARCH_NAME"
echo "Simulator Path      : $TT_SIM_LIB_PATH"
echo "Framework Root      : $TT_METAL_HOME"

if [ -f "$TT_SIM_LIB_PATH" ]; then
    echo "Simulator Binary    : ✅ Found"
else
    echo "Simulator Binary    : ❌ NOT FOUND (Expected at $TT_SIM_LIB_PATH)"
fi

if [ -d "$TT_METAL_HOME/build/lib" ]; then
    echo "Build Libraries     : ✅ Found"
else
    echo "Build Libraries     : ⚠️ Missing build/lib folder. Ensure tt-metal is built."
fi
echo "=================================================="
