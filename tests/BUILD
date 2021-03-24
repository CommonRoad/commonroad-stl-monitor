py_test(
    name = "test_rtamt",
    srcs = ["test_rtamt.py"],
    deps = ["@rtamt_repo//:rtamt"]
)

py_test(
    name = "test_mtl_stl_equivalence",
    srcs = ["test_mtl_stl_equivalence.py"],
    data = ["//scenarios:test_scenarios"],
    deps = ["//crmonitor"]
)
