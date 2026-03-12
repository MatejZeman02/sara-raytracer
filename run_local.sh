#!/bin/bash

# clear

# get absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
MAIN_SCRIPT="$PROJECT_ROOT/src/main.py"
TINYOBJ_SO="$PROJECT_ROOT/utils/tinyobjloader_py.so"

# Check if the file exists
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "Error: File $MAIN_SCRIPT not found!"
    exit 1
fi

# Check if tinyobjloader_py.so is compiled
if [ ! -f "$TINYOBJ_SO" ]; then
    echo "Warning: $TINYOBJ_SO not found!"
    echo "Please compile it using: ./rebuild_tinyobjloader.sh (or CMAKE)."
fi

conda run -n raytracer python "$MAIN_SCRIPT" || { echo "NOTE: this script requires conda environment 'raytracer'"; exit 1; }
