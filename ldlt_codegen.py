import numpy as np
import scipy as sp


def _build_sparse_L(M, PINV):
    n = M.shape[0]

    P_inv = np.zeros_like(M)
    P_inv[np.arange(n), PINV] = 1.0

    P = P_inv.T

    N = P @ M @ P_inv

    L = (np.tril(N, k=-1) != 0.0)

    # Note that The L D L^T decomposition of N can be computed with the following recursion:
    # D_i    = N_ii - sum_{j=0}^{i-1} L_{ij}^2 D_j
    # L_{ij} = (1 / D_j) * (N_{ij} - sum_{k=0}^{j-1} L_{ik} L_{jk} D_k)

    for i in range(n):
        for j in range(i):
            L[i, j] = L[i, j] or np.any(np.logical_and(L[i, :j], L[j, :j]))

    return sp.sparse.csc_matrix(L)


def ldlt_codegen(M, PINV, namespace, header_name):
    dim = M.shape[0]
    SPARSE_UPPER_M = sp.sparse.csc_matrix(np.triu(M))

    SPARSE_L = _build_sparse_L(M=M, PINV=PINV)

    L_nnz = SPARSE_L.nnz

    # Maps (i, j) and (j, i), representing indices of PERMUTED_M,
    # to the data-index of the CSC representation of
    # SPARSE_UPPER_M[PINV[min(i, j)], PINV[max(i, j)]], where PINV = PINV.
    # Note: PERMUTED_M[i, j] = M[PINV[i], PINV[j]], where PINV = PINV.
    PM_COORDINATE_MAP = {}
    for i in range(dim):
        for j in range(
            SPARSE_UPPER_M.indptr[i],
            SPARSE_UPPER_M.indptr[i + 1],
        ):
            k = SPARSE_UPPER_M.indices[j]
            ii = int(PINV[min(i, k)])
            kk = int(PINV[max(i, k)])
            PM_COORDINATE_MAP[(ii, kk)] = j
            PM_COORDINATE_MAP[(kk, ii)] = j

    L_COORDINATE_MAP = {}
    L_nz_per_row = [set() for _ in range(dim)]
    L_nz_per_col = [set() for _ in range(dim)]
    for j in range(dim):
        for k in range(SPARSE_L.indptr[j], SPARSE_L.indptr[j + 1]):
            i = int(SPARSE_L.indices[k])
            assert i > j
            L_COORDINATE_MAP[(i, j)] = k
            L_nz_per_row[i].add(j)
            L_nz_per_col[j].add(i)

    ldlt_impl = ""

    for i in range(dim):
        for j in L_nz_per_row[i]:
            assert (i, j) in L_COORDINATE_MAP
            L_idx = L_COORDINATE_MAP[(i, j)]
            line = f"L_data[{L_idx}] = ("
            if (i, j) in PM_COORDINATE_MAP:
                A_idx = PM_COORDINATE_MAP[(i, j)]
                line += f"A_data[{A_idx}]"
            for k in range(j):
                if (i, k) not in L_COORDINATE_MAP or (j, k) not in L_COORDINATE_MAP:
                    continue
                line += f" - L_data[{L_COORDINATE_MAP[(i, k)]}] * L_data[{L_COORDINATE_MAP[(j, k)]}] * D_diag[{k}]"
            line += f") / D_diag[{j}];\n"
            ldlt_impl += line

        # Update D_diag.
        line = f"D_diag[{i}] = "
        if (i, i) in PM_COORDINATE_MAP:
            line += f"A_data[{PM_COORDINATE_MAP[(i, i)]}]"
        for j in L_nz_per_row[i]:
            assert (i, j) in L_COORDINATE_MAP
            L_data_ij_idx = L_COORDINATE_MAP[(i, j)]
            line += f" - L_data[{L_data_ij_idx}] * L_data[{L_data_ij_idx}] * D_diag[{j}]"
        line += ";\n"
        ldlt_impl += line

    solve_lower_unitriangular_impl = ""

    for i in range(dim):
        line = f"x[{i}] = b[{i}] "
        for j in L_nz_per_row[i]:
            assert (i, j) in L_COORDINATE_MAP
            line += f"- L_data[{L_COORDINATE_MAP[(i, j)]}] * x[{j}]"
        line += ";\n"
        solve_lower_unitriangular_impl += line

    solve_upper_unitriangular_impl = ""

    for i in range(dim - 1, -1, -1):
        line = f"x[{i}] = b[{i}] "
        for j in L_nz_per_col[i]:
            assert (j, i) in L_COORDINATE_MAP
            line += f"- L_data[{L_COORDINATE_MAP[(j, i)]}] * x[{j}]"
        line += ";\n"
        solve_upper_unitriangular_impl += line

    permute_b = ""
    for i in range(dim):
        permute_b += f"tmp2[{PINV[i]}] = b[{i}];\n"

    permute_solution = ""
    for i in range(dim):
        permute_solution += f"x[{i}] = tmp2[{PINV[i]}];\n"

    cpp_header_code = f"""
    #pragma once

    namespace {namespace}
    {{
    // Performs an L D L^T decomposition of the A matrix,
    // where A_data is expected to represent np.triu(A) in CSC order.
    void ldlt(const double* A_data, double* L_data, double* D_diag);

    // Solves (L + I) x = b for x, where L is strictly lower triangular,
    // and where L_data is expected to represent L in CSC order.
    void solve_lower_unitriangular(const double* L_data, const double* b, double* x);

    // Solves (L + I).T x = b for x, where L is strictly lower triangular,
    // and where L_data is expected to represent L in CSC order.
    void solve_upper_unitriangular(const double* L_data, const double* b, double* x);

    // Solves A * x = b via an L D L^T decomposition,
    // where A_data is expected to represent np.triu(A) in CSC order.
    void ldlt_solve(const double* A_data, const double* b, double* x);
    }}  // namespace {namespace}
    """

    cpp_impl_code = f"""#include "{header_name}.hpp"

    #include <array>

    namespace {namespace}
    {{

    void ldlt(const double* A_data, double* L_data, double* D_diag) {{
    {ldlt_impl}
    }}

    void solve_lower_unitriangular(const double* L_data, const double* b, double* x) {{
    {solve_lower_unitriangular_impl}
    }}

    void solve_upper_unitriangular(const double* L_data, const double* b, double* x) {{
    {solve_upper_unitriangular_impl}
    }}

    void ldlt_solve(const double* A_data, const double* b, double* x) {{
    std::array<double, {L_nnz}> L_data;
    std::array<double, {dim}> D_diag;
    std::array<double, {dim}> tmp1;
    std::array<double, {dim}> tmp2;
    ldlt(A_data, L_data.data(), D_diag.data());
    {permute_b}
    solve_lower_unitriangular(L_data.data(), tmp2.data(), tmp1.data());
    for (std::size_t i = 0; i < {dim}; ++i) {{
        tmp1[i] /= D_diag[i];
    }}
    solve_upper_unitriangular(L_data.data(), tmp1.data(), tmp2.data());
    {permute_solution}
    }}

    }} // namespace {namespace}
    """

    return cpp_header_code, cpp_impl_code
