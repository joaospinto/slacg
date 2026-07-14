import sys

import numpy as np
import scipy as sp
from slacg.kkt_codegen import kkt_codegen, write_generated_files

x_dim = 2
y_dim = 1
z_dim = 0
dim = x_dim + y_dim + z_dim

H = sp.sparse.eye(x_dim, format="csc")
C = sp.sparse.csc_matrix(np.ones((y_dim, x_dim)))
G = sp.sparse.csc_matrix((z_dim, x_dim))
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
        num_solve_chunks=2,
    ),
)
