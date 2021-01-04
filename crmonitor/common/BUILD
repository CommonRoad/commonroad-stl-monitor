py_library(
    name = "common",
    srcs = glob(["*.py"]),
    data = ["//crmonitor/common/cpp:crmonitor_cpp.so"],
    imports = ["cpp"],
    visibility = ["//visibility:public"],
    deps = ["@crccosy//:pycrccosy_lib"],
)
