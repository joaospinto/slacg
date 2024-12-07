load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "common",
  srcs = ["common.py"],
  deps = [
    requirement("numpy"),
    requirement("scipy"),
  ],
  visibility = ["//visibility:public"],
)
