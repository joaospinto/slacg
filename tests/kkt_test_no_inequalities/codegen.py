import sys

import numpy as np
import scipy as sp
from slacg.kkt_codegen import kkt_codegen

x_dim = 2
y_dim = 1
z_dim = 0
dim = x_dim + y_dim + z_dim

H = sp.sparse.eye(x_dim, format="csc")
C = sp.sparse.csc_matrix(np.ones((y_dim, x_dim)))
G = sp.sparse.csc_matrix((z_dim, x_dim))
P = np.arange(dim - 1, -1, -1)

output_prefix = sys.argv[-1]
cpp_header_code, cpp_impl_code = kkt_codegen(
    H=H,
    C=C,
    G=G,
    P=P,
    namespace="slacg::test",
    header_name="kkt_codegen",
)

with open(f"{output_prefix}/kkt_codegen.hpp", "w") as f:
    f.write(cpp_header_code)

with open(f"{output_prefix}/kkt_codegen.cpp", "w") as f:
    f.write(cpp_impl_code)
