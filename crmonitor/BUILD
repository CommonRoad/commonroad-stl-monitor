py_library(
    name = "crmonitor",
    srcs = glob(["**/*.py"]),
    data = ["//crmonitor:data_files"],
    visibility = ["//visibility:public"],
    deps = [
        "//crmonitor/common",
        "@crccosy//:pycrccosy_lib",
    ],
)

filegroup(
    name = "data_files",
    srcs = glob(["**/*.yaml"]),
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_import",
    srcs = ["test_import.py"],
    deps = [":crmonitor"],
)
