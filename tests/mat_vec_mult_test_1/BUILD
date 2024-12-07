load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
  name = "codegen",
  srcs = ["codegen.py"],
  deps = [
    "//slacg:mat_vec_mult_codegen"
  ]
)

genrule(
  name="codegen_genrule",
  outs = [
    "mat_vec_mult_codegen.hpp",
    "mat_vec_mult_codegen.cpp",
  ],
  cmd = "$(location :codegen) $(RULEDIR) > /dev/null",
  tools = [":codegen"],
)

cc_library(
  name = "codegen_output",
  srcs = [
    "mat_vec_mult_codegen.cpp",
  ],
  hdrs = [
    "mat_vec_mult_codegen.hpp",
  ],
)

cc_test(
  name = "test",
  size = "small",
  srcs = [
    "test.cpp",
  ],
  deps = [
    "@googletest//:gtest",
    "@googletest//:gtest_main",
    ":codegen_output"
  ],
  visibility = ["//visibility:public",],
)
