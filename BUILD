load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "ldlt_codegen",
  srcs = ["ldlt_codegen.py"],
  deps = [
    requirement("numpy"),
    requirement("scipy"),
  ],
  visibility = ["//visibility:public"],
)

py_binary(
      name = "codegen_for_example",
      srcs = ["codegen_for_example.py"],
      deps = [":ldlt_codegen"]
)

genrule(
  name="codegen_for_example_genrule",
  outs = [
    "ldlt_codegen_for_example.hpp",
    "ldlt_codegen_for_example.cpp",
  ],
  cmd = "$(location :codegen_for_example) $(RULEDIR) > /dev/null",
  tools = [":codegen_for_example"],
)

cc_library(
  name = "example_codegen",
  srcs = [
    "ldlt_codegen_for_example.cpp",
  ],
  hdrs = [
    "ldlt_codegen_for_example.hpp",
  ],
)

cc_test(
  name = "example",
  size = "small",
  srcs = [
    "example.cpp",
  ],
  deps = [
    "@googletest//:gtest",
    "@googletest//:gtest_main",
    ":example_codegen"
  ],
  visibility = ["//visibility:public",],
)
