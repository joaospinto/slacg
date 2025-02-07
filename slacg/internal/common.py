import numpy as np
import scipy as sp

def build_sparse_LT(M, P):
    n = M.shape[0]

    P_MAT = np.zeros_like(M)
    P_MAT[np.arange(n), P] = 1.0

    N = P_MAT @ M @ P_MAT.T
    U = N != 0

    nonzero_rows_per_col = [set() for _ in range(U.shape[1])]
    for col in range(U.shape[1]):
        for row in range(col):
            if U[row, col] or nonzero_rows_per_col[row].intersection(nonzero_rows_per_col[col]):
                nonzero_rows_per_col[col].add(row)

    csc_rows = []
    csc_cols = []
    csc_values = []

    for col, row_set in enumerate(nonzero_rows_per_col):
        if row_set:
            csc_rows.extend(list(row_set))
            csc_cols.extend([col] * len(row_set))
            csc_values.extend([True] * len(row_set))

    return sp.sparse.csc_matrix((csc_values, (csc_rows, csc_cols)), shape=U.shape)
