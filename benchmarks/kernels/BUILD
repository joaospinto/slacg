load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
  name = "codegen",
  srcs = ["codegen.py"],
  deps = [
    "//slacg:gtwg_codegen",
    "//slacg:kkt_codegen",
    "//slacg:ldlt_codegen",
    "//slacg:mat_vec_mult_codegen",
  ],
)

genrule(
  name = "codegen_genrule",
  outs = [
    "gtwg_codegen.cpp",
    "gtwg_codegen.hpp",
    "kkt_codegen.cpp",
    "kkt_codegen.hpp",
    "ldlt_codegen.cpp",
    "ldlt_codegen.hpp",
    "mat_vec_mult_codegen.cpp",
    "mat_vec_mult_codegen.hpp",
  ],
  cmd = "$(location :codegen) $(RULEDIR) > /dev/null",
  tools = [":codegen"],
)

cc_library(
  name = "codegen_output",
  srcs = [
    "gtwg_codegen.cpp",
    "kkt_codegen.cpp",
    "ldlt_codegen.cpp",
    "mat_vec_mult_codegen.cpp",
  ],
  hdrs = [
    "gtwg_codegen.hpp",
    "kkt_codegen.hpp",
    "ldlt_codegen.hpp",
    "mat_vec_mult_codegen.hpp",
  ],
)

cc_binary(
  name = "benchmark",
  srcs = ["benchmark.cpp"],
  deps = [":codegen_output"],
  visibility = ["//visibility:public"],
)
