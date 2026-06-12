import numpy as np
import scipy as sp

import sys

from slacg.ldlt_codegen import ldlt_codegen
from slacg.mat_vec_mult_codegen import mat_vec_mult_codegen

x_dim = 10
y_dim = 20
s_dim = 30

dim = x_dim + y_dim + s_dim

P_mat = sp.sparse.eye(x_dim, format="csc")
C = sp.sparse.csc_matrix(np.ones([y_dim, x_dim]))
G = sp.sparse.csc_matrix(np.ones([s_dim, x_dim]))
I_y = sp.sparse.eye(y_dim, format="csc")
W = sp.sparse.eye(s_dim, format="csc")
Zys = sp.sparse.csc_matrix((y_dim, s_dim))

M = sp.sparse.bmat(
    [
        [P_mat, C.T, G.T],
        [C, -I_y, Zys],
        [G, Zys.T, -W],
    ],
    format="csc",
)

P = np.arange(dim - 1, -1, -1)

output_prefix = sys.argv[-1]

cpp_header_code, cpp_impl_code = ldlt_codegen(
    M=M, P=P, namespace="slacg::test", header_name="ldlt_codegen"
)

with open(f"{output_prefix}/ldlt_codegen.hpp", "w") as f:
    f.write(cpp_header_code)

with open(f"{output_prefix}/ldlt_codegen.cpp", "w") as f:
    f.write(cpp_impl_code)

cpp_header_code, cpp_impl_code = mat_vec_mult_codegen(
    M=M, namespace="slacg::test", header_name="mat_vec_mult_codegen"
)

with open(f"{output_prefix}/mat_vec_mult_codegen.hpp", "w") as f:
    f.write(cpp_header_code)

with open(f"{output_prefix}/mat_vec_mult_codegen.cpp", "w") as f:
    f.write(cpp_impl_code)
