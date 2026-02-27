cmake_minimum_required(VERSION 3.15)
project(tinyobjloader_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

add_library(tinyobjloader_py MODULE
    py_tinyobj.cpp
    TinyObjLoader.h
)

target_include_directories(tinyobjloader_py
    PRIVATE ${pybind11_INCLUDE_DIRS}
)

# Required for Python extension modules
set_target_properties(tinyobjloader_py PROPERTIES
    PREFIX ""
)
