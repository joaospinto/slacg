import numpy as np

import sys

from ldlt_codegen import ldlt_codegen

H = np.arange(9).reshape([3, 3])
M = H.T @ H + np.eye(3)

PINV = np.array([2, 0, 1])

cpp_header_code, cpp_impl_code = ldlt_codegen(
        M=M,
        PINV=PINV,
        namespace="slacg::example",
        header_name="ldlt_codegen_for_example"
)

output_prefix = sys.argv[-1]

with open(f"{output_prefix}/ldlt_codegen_for_example.hpp", "w") as f:
    f.write(cpp_header_code)

with open(f"{output_prefix}/ldlt_codegen_for_example.cpp", "w") as f:
    f.write(cpp_impl_code)
