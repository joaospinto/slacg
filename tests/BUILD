load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
  name = "codegen_for_example",
  srcs = ["codegen_for_example.py"],
  deps = [
    "//src/slacg:ldlt_codegen",
    "//src/slacg:mat_vec_mult_codegen"
  ]
)

genrule(
  name="codegen_for_example_genrule",
  outs = [
    "ldlt_codegen_for_example.hpp",
    "ldlt_codegen_for_example.cpp",
    "mat_vec_mult_codegen_for_example.hpp",
    "mat_vec_mult_codegen_for_example.cpp",
  ],
  cmd = "$(location :codegen_for_example) $(RULEDIR) > /dev/null",
  tools = [":codegen_for_example"],
)

cc_library(
  name = "example_codegen",
  srcs = [
    "ldlt_codegen_for_example.cpp",
    "mat_vec_mult_codegen_for_example.cpp",
  ],
  hdrs = [
    "ldlt_codegen_for_example.hpp",
    "mat_vec_mult_codegen_for_example.hpp",
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
