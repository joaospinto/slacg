import numpy as np
import scipy as sp

import sys

from slacg.mat_vec_mult_codegen import mat_vec_mult_codegen

M = sp.sparse.csc_matrix(np.arange(5 * 10).reshape([5, 10]))

output_prefix = sys.argv[-1]

cpp_header_code, cpp_impl_code = mat_vec_mult_codegen(
    M=M, namespace="slacg::test", header_name="mat_vec_mult_codegen"
)

with open(f"{output_prefix}/mat_vec_mult_codegen.hpp", "w") as f:
    f.write(cpp_header_code)

with open(f"{output_prefix}/mat_vec_mult_codegen.cpp", "w") as f:
    f.write(cpp_impl_code)
