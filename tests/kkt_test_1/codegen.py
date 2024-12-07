import numpy as np

import sys

from src.slacg.kkt_codegen import kkt_codegen

x_dim = 10
y_dim = 20
s_dim = 30

dim = x_dim + y_dim + s_dim

H = np.eye(x_dim)
C = np.ones([y_dim, x_dim])
G = np.ones([s_dim, x_dim])
I_y = np.eye(y_dim)
Sigma_inv = np.eye(s_dim)
Zys = np.zeros([y_dim, s_dim])

P = np.arange(dim - 1, -1, -1)

output_prefix = sys.argv[-1]

cpp_header_code, cpp_impl_code = kkt_codegen(
    H=H, C=C, G=G, P=P, namespace="slacg::test", header_name="kkt_codegen"
)

with open(f"{output_prefix}/kkt_codegen.hpp", "w") as f:
    f.write(cpp_header_code)

with open(f"{output_prefix}/kkt_codegen.cpp", "w") as f:
    f.write(cpp_impl_code)
