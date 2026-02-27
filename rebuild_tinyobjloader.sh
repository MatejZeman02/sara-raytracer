#!/bin/bash
rm -rf utils/build

# CMake files pybind11 from Python
PYBIND11_CMAKE_DIR=$(python3.14 -m pybind11 --cmakedir)

# Pass this path to CMake via -Dpybind11_DIR
cmake -S utils -B utils/build -Dpybind11_DIR="$PYBIND11_CMAKE_DIR"

cmake --build utils/build

# Warning: Pybind11 generates a file with a suffix (e.g. .cpython-314-x86_64-linux-gnu.so)
# use wildcard * to make it work
mv utils/build/tinyobjloader_py*.so utils/tinyobjloader_py.so

rm -rf utils/build
