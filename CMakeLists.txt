cmake_minimum_required(VERSION 2.8.12)
project(crmonitor)

add_subdirectory(external/pybind11)
pybind11_add_module(crmonitor crmonitor/common/cpp/helper.cpp)