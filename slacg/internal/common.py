import numpy as np
import scipy as sp

RESTRICT_MACRO = """#if defined(__GNUC__) || defined(__clang__)
#define SLACG_RESTRICT __restrict__
#else
#define SLACG_RESTRICT
#endif

// Pointer arguments marked SLACG_RESTRICT must not point to overlapping memory.

"""


def build_sparse_LT(M, P):
    assert sp.sparse.issparse(M)

    n = M.shape[0]
    M = M.tocsc(copy=True)
    M.eliminate_zeros()
    P = np.asarray(P, dtype=int)

    N = M.tocsr()[P, :][:, P].tocsc()
    U = (N != 0).tocsc()

    nonzero_rows_per_input_col = [set() for _ in range(U.shape[1])]
    for col in range(U.shape[1]):
        rows = U.indices[U.indptr[col] : U.indptr[col + 1]]
        nonzero_rows_per_input_col[col] = {int(row) for row in rows if row < col}

    nonzero_rows_per_col = [set() for _ in range(U.shape[1])]
    for col in range(U.shape[1]):
        input_col_rows = nonzero_rows_per_input_col[col]
        for row in range(col):
            if row in input_col_rows or nonzero_rows_per_col[row].intersection(
                nonzero_rows_per_col[col]
            ):
                nonzero_rows_per_col[col].add(row)

    csc_rows = []
    csc_cols = []
    csc_values = []

    for col, row_set in enumerate(nonzero_rows_per_col):
        if row_set:
            rows = sorted(row_set)
            csc_rows.extend(rows)
            csc_cols.extend([col] * len(row_set))
            csc_values.extend([True] * len(row_set))

    return sp.sparse.csc_matrix((csc_values, (csc_rows, csc_cols)), shape=U.shape)
