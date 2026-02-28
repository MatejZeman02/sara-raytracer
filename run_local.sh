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

# Check if the file exists
if [ ! -f "$HW_SCRIPT" ]; then
    echo "Error: File $HW_SCRIPT not found!"
    exit 1
fi

conda run -n raytracer python "$HW_SCRIPT" || { echo "\nError in $HW_SCRIPT"; exit 1; }
