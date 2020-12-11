py_library(
	name="crmonitor",
	srcs = glob(["**/*.py"]),
	visibility = ["//visibility:public"],
	deps=["//crmonitor/common:common", "@curvilinear//:pycrccosy_lib"],
	data = ["@stl_crmonitor//crmonitor/common/cpp:crmonitor_cpp.so", "@stl_crmonitor//crmonitor:data_files",],
)

filegroup(
	name="data_files",
	srcs = glob(["**/*.yaml"]),
	visibility = ["//visibility:public"],
)