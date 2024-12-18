load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
  name = "codegen",
  srcs = ["codegen.py"],
  deps = [
    "//slacg:gtwg_codegen"
  ]
)

genrule(
  name="codegen_genrule",
  outs = [
    "gtwg_codegen.hpp",
    "gtwg_codegen.cpp",
  ],
  cmd = "$(location :codegen) $(RULEDIR) > /dev/null",
  tools = [":codegen"],
)

cc_library(
  name = "codegen_output",
  srcs = [
    "gtwg_codegen.cpp",
  ],
  hdrs = [
    "gtwg_codegen.hpp",
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
