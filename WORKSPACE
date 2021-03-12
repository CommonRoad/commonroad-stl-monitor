workspace(name = "stl_crmonitor")

load("//tools:repositories.bzl", "load_stlmonitor_repos")

load_stlmonitor_repos()

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(name = "local_config_python")

load("@crccosy//tools:repositories.bzl", "load_crccosy_repos")

load_crccosy_repos()

# Experimental for including rtamt
#all_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""

#
#load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")
#
#rules_foreign_cc_dependencies(
#    ["//:built_cmake_toolchain"],
#    #register_default_tools = False,
#)

#new_git_repository(
#    name = "rtamt",
#    branch = "master",
#    build_file_content = all_content,
#    remote = "https://github.com/nickovic/rtamt.git",
#)

#new_local_repository(
#    name = "rtamt",
#    build_file_content = all_content,
#    path = "/home/luis/Documents/Promotion/Code/rtamt",
#)
