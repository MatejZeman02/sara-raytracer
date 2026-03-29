#!/bin/bash
set -e

# get Python from argument or default to current environment's python
PYTHON_CMD="${1:-python}"

# Verify Python is available
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "Error: $PYTHON_CMD not found"
    exit 1
fi


# CMake files pybind11 from Python:
# Get the CMake directory for pybind11 using Python 3.14's pip module system.
# This command queries pybind11's installation path to find its CMake configuration files,
# which are needed for building C++ extensions with pybind11.
#
# Note: Using 'python3.14' explicitly specifies Python version 3.14. This could be replaced
# with 'python3' to use the system's default Python 3 version, unless you specifically need
# Python 3.14's pybind11 installation (e.g., when multiple Python versions are installed
# and you're building an extension for Python 3.14 specifically).
PYBIND11_CMAKE_DIR=$("$PYTHON_CMD" -m pybind11 --cmakedir)

# pass path to CMake via -Dpybind11_DIR
cmake -S utils -B utils/build -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE="$(command -v "$PYTHON_CMD")"

cmake --build utils/build

# Warning: Pybind11 generates a file with a suffix (e.g. .cpython-314-x86_64-linux-gnu.so)
# use wildcard * to make it work
mv utils/build/tinyobjloader_py*.so utils/tinyobjloader_py.so

rm -rf utils/build
