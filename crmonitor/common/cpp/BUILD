load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "helper",
    srcs = ["helper.cpp"],
)

py_test(
    name = "test_helper",
    srcs = ["test_helper.py"],
    data = [":helper.so"],
)