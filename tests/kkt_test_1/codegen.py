import numpy as np
import scipy as sp

import sys

from slacg.kkt_codegen import kkt_codegen, write_generated_files

x_dim = 10
y_dim = 20
s_dim = 30

dim = x_dim + y_dim + s_dim

H = sp.sparse.eye(x_dim, format="csc")
C = sp.sparse.csc_matrix(np.ones([y_dim, x_dim]))
G = sp.sparse.csc_matrix(np.ones([s_dim, x_dim]))

P = np.arange(dim - 1, -1, -1)

output_prefix = sys.argv[-1]

write_generated_files(
    output_prefix,
    kkt_codegen(
        H=H,
        C=C,
        G=G,
        P=P,
        namespace="slacg::test",
        header_name="kkt_codegen",
        bordered_x_indices=(x_dim - 1,),
        num_factor_chunks=2,
        num_product_chunks=4,
        num_solve_chunks=3,
    ),
)
