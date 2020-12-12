py_library(
	name="common",
	srcs=glob(["*.py"]),
	visibility = ["//visibility:public"],
	imports = ["cpp"],
	data=["@stl_crmonitor//crmonitor/common/cpp:crmonitor_cpp.so",],
	deps = ["@curvilinear//:pycrccosy_lib"]
)

py_test(
    name="test_import",
    srcs=["test_import.py"],
    deps=[":common"]
)