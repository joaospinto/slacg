import sys

import numpy as np

from slacg.gtwg_codegen import gtwg_codegen
from slacg.kkt_codegen import kkt_codegen
from slacg.ldlt_codegen import ldlt_codegen
from slacg.mat_vec_mult_codegen import mat_vec_mult_codegen


def banded_spd(dim, half_bandwidth):
    M = np.eye(dim)
    for offset in range(1, half_bandwidth + 1):
        value = 1.0 / (offset + 1)
        M += np.diag(np.full(dim - offset, value), offset)
        M += np.diag(np.full(dim - offset, value), -offset)
    return M


def sparse_pattern(rows, cols, entries_per_col):
    M = np.zeros((rows, cols))
    for col in range(cols):
        for offset in range(entries_per_col):
            row = (3 * col + 5 * offset) % rows
            M[row, col] = 1.0
    return M


output_prefix = sys.argv[-1]

ldlt_dim = 36
ldlt_M = banded_spd(ldlt_dim, 4)
ldlt_P = np.arange(ldlt_dim - 1, -1, -1)

matvec_M = sparse_pattern(72, 48, 5)

gtwg_G = sparse_pattern(64, 32, 6)

x_dim = 24
y_dim = 16
z_dim = 32
H = banded_spd(x_dim, 3)
C = sparse_pattern(y_dim, x_dim, 4)
G = sparse_pattern(z_dim, x_dim, 5)
kkt_P = np.arange(x_dim + y_dim + z_dim - 1, -1, -1)

outputs = {
    "ldlt_codegen": ldlt_codegen(
        M=ldlt_M,
        P=ldlt_P,
        namespace="slacg::bench::ldlt",
        header_name="benchmarks/kernels/ldlt_codegen",
    ),
    "mat_vec_mult_codegen": mat_vec_mult_codegen(
        M=matvec_M,
        namespace="slacg::bench::matvec",
        header_name="benchmarks/kernels/mat_vec_mult_codegen",
    ),
    "gtwg_codegen": gtwg_codegen(
        G=gtwg_G,
        namespace="slacg::bench::gtwg",
        header_name="benchmarks/kernels/gtwg_codegen",
    ),
    "kkt_codegen": kkt_codegen(
        H=H,
        C=C,
        G=G,
        P=kkt_P,
        namespace="slacg::bench::kkt",
        header_name="benchmarks/kernels/kkt_codegen",
    ),
}

for name, (cpp_header_code, cpp_impl_code) in outputs.items():
    with open(f"{output_prefix}/{name}.hpp", "w") as f:
        f.write(cpp_header_code)

    with open(f"{output_prefix}/{name}.cpp", "w") as f:
        f.write(cpp_impl_code)
