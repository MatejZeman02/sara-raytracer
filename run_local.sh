#!/bin/bash

# clear

HW="hw02"
# check if argument is provided
if [ $# -gt 0 ]; then
    HW="$1"
fi

# get absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
HW_SCRIPT="$PROJECT_ROOT/$HW/main.py"
TINYOBJ_SO="$PROJECT_ROOT/utils/tinyobjloader_py.so"

# Check if the file exists
if [ ! -f "$HW_SCRIPT" ]; then
    echo "Error: File $HW_SCRIPT not found!"
    exit 1
fi

# Check if tinyobjloader_py.so is compiled
if [ ! -f "$TINYOBJ_SO" ]; then
    echo "Warning: $TINYOBJ_SO not found!"
    echo "Please compile it using: ./rebuild_tinyobjloader.sh (or CMAKE)."
fi

# NUMBA_ENABLE_CUDASIM=1 # for debugging in CPU mode
conda run -n raytracer python "$HW_SCRIPT" || { echo "\nError in $HW_SCRIPT"; exit 1; }
