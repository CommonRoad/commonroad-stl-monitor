#toolchain(
#    name = "built_cmake_toolchain",
#    exec_compatible_with = [
#        "@bazel_tools//platforms:osx",
#        "@bazel_tools//platforms:x86_64",
#    ],
#    toolchain = "@rules_foreign_cc//tools/build_defs/native_tools:built_cmake",
#    toolchain_type = "@rules_foreign_cc//tools/build_defs:cmake_toolchain",
#)
#
#load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")
#
#cmake_external(
#    name = "rtamt",
#    #    additional_inputs = ["postfix_rtamt.sh"],
#    # Values to be passed as -Dkey=value on the CMake command line;
#    # here are serving to provide some CMake script configuration options
#    cache_entries = {
#        "PythonVersion": "3",
#    },
#    lib_source = "@rtamt//:all",
#    make_commands = ["make"],
#    #    postfix_script = "pwd",
#    out_lib_dir = "cpplib/stl/rtamt_stl_library",
#    # We are selecting the resulting static library to be passed in C/C++ provider
#    # as the result of the build;
#    # However, the cmake_external dependants could use other artefacts provided by the build,
#    # according to their CMake script
#    shared_libraries = ["librtamt_stl_library.so"],
#    #   working_directory = "rtamt"
#)
