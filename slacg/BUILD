load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "kkt_codegen",
  srcs = [
    "__init__.py",
    "kkt_codegen.py",
  ],
  deps = [
    requirement("numpy"),
    requirement("scipy"),
    "//slacg/internal:common",
  ],
  visibility = ["//visibility:public"],
)

py_library(
  name = "ldlt_codegen",
  srcs = [
    "__init__.py",
    "ldlt_codegen.py",
  ],
  deps = [
    requirement("numpy"),
    requirement("scipy"),
    "//slacg/internal:common",
  ],
  visibility = ["//visibility:public"],
)

py_library(
  name = "mat_vec_mult_codegen",
  srcs = [
    "__init__.py",
    "mat_vec_mult_codegen.py",
  ],
  deps = [
    requirement("numpy"),
    requirement("scipy"),
  ],
  visibility = ["//visibility:public"],
)

py_library(
  name = "gtwg_codegen",
  srcs = [
    "__init__.py",
    "gtwg_codegen.py",
  ],
  deps = [
    requirement("numpy"),
    requirement("scipy"),
  ],
  visibility = ["//visibility:public"],
)
