load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def load_stlmonitor_repos():

    maybe(
        http_archive,
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-26973c0ff320cb4b39e45bc3e4297b82bc3a6c09",
        urls = ["https://github.com/pybind/pybind11_bazel/archive/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.zip"],
        sha256 = "a5666d950c3344a8b0d3892a88dc6b55c8e0c78764f9294e806d69213c03f19d",
    )

    maybe(
        http_archive,
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        strip_prefix = "pybind11-2.6.1",
        sha256 = "cdbe326d357f18b83d10322ba202d69f11b2f49e2d87ade0dc2be0c5c34f8e2a",
        urls = ["https://github.com/pybind/pybind11/archive/v2.6.1.tar.gz"],
    )

    maybe(
        git_repository,
        name = "crccosy",
        commit = "6b4b290c0c323ce16e583b8f59786da438249743",
        remote = "git@gitlab.lrz.de:cps/commonroad-curvilinear-coordinate-system.git",
        shallow_since = "1608644996 +0100",
    )

    maybe(
        new_git_repository,
        name = "rtamt_repo",
        commit = "5660f315d77f62b2b74d1c65ff9b48ee17afb5a1",
        build_file_content = """py_library(name = "rtamt",srcs = glob(["rtamt/**/*.py"]),visibility = ["//visibility:public"], imports = ["rtamt"])""",
        remote = "https://github.com/cirrostratus1/rtamt.git",
    )
