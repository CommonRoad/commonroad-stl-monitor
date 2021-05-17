workspace(name = "stl_crmonitor")

load("//tools:repositories.bzl", "load_stlmonitor_repos")

load_stlmonitor_repos()

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(name = "local_config_python")

load("@crccosy//tools:repositories.bzl", "load_crccosy_repos")

load_crccosy_repos()