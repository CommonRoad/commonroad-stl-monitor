load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "crmonitor_cpp",
    srcs = ["helper.cpp"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_helper",
    srcs = ["test_helper.py"],
    data = [":crmonitor_cpp.so"],
)