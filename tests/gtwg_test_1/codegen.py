import numpy as np
import scipy as sp

import sys

from slacg.gtwg_codegen import gtwg_codegen

G = sp.sparse.csc_matrix(np.arange(5 * 10).reshape([5, 10]))

output_prefix = sys.argv[-1]

cpp_header_code, cpp_impl_code = gtwg_codegen(
    G=G, namespace="slacg::test", header_name="gtwg_codegen"
)

with open(f"{output_prefix}/gtwg_codegen.hpp", "w") as f:
    f.write(cpp_header_code)

with open(f"{output_prefix}/gtwg_codegen.cpp", "w") as f:
    f.write(cpp_impl_code)
