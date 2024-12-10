import numpy as np
import scipy as sp

from slacg.internal.common import build_sparse_LT

# TODO(joao): implement the same changes (ie L -> LT) here!


# NOTE:
# 1. Given a permutation P, we define the associated permutation matrix
#    P_MAT as P_MAT[i, j] = 0 iff j = p[i]. For example, if P = (2, 0, 1),
#    P_MAT = [[0, 0, 1],
#             [1, 0, 0],
#             [0, 1, 0]].
# 2. The user should pass a permutation P for which (P_MAT @ M @ P_MAT.T)
#    has as little fill-in as possible in its L D L^T decomposition.
#    Typically, an AMD ordering is pre-computed for the sparsity pattern.


def ldlt_codegen(M, P, namespace, header_name):
    dim = M.shape[0]
    SPARSE_UPPER_M = sp.sparse.csc_matrix(np.triu(M))

    SPARSE_LT = build_sparse_LT(M=M, P=P)

    L_nnz = SPARSE_LT.nnz

    P_MAT = np.zeros_like(M)
    P_MAT[np.arange(dim), P] = 1.0

    N = P_MAT @ M @ P_MAT.T

    # NOTE:
    # 1. P_MAT[i, j] = 0 iff j = P[i]
    # 2. N[i, j] = (P_MAT M P_MAT.T)[i, j] = sum_k (P_MAT M)[i, k] (P_MAT.T)[k, j]
    #            = sum_k (P_MAT M)[i, k] P_MAT[j, k] = (P_MAT M)[i, P[j]]
    #            = sum_k P_MAT[i, k] M[k, P[j]] = M[P[i], P[j]]
    # 3. M[i, j] = N[PINV[i], PINV[j]]

    PINV = np.zeros_like(P)
    PINV[P] = np.arange(dim)

    # N_COORDINATE_MAP maps (m, n) and (n, m), representing indices of N,
    # to the data-index of the CSC representation of
    # SPARSE_UPPER_M[minmax(PINV[m], PINV[n])].

    N_COORDINATE_MAP = {}
    for j in range(dim):
        for k in range(
            SPARSE_UPPER_M.indptr[j],
            SPARSE_UPPER_M.indptr[j + 1],
        ):
            i = int(SPARSE_UPPER_M.indices[k])
            m = int(PINV[i])
            n = int(PINV[j])
            assert N[m, n] == M[i, j]
            N_COORDINATE_MAP[(m, n)] = k
            N_COORDINATE_MAP[(n, m)] = k

    L_COORDINATE_MAP = {}
    L_nz_set_per_row = [set() for _ in range(dim)]
    L_nz_set_per_col = [set() for _ in range(dim)]
    for i in range(dim):
        for k in range(SPARSE_LT.indptr[i], SPARSE_LT.indptr[i + 1]):
            j = int(SPARSE_LT.indices[k])
            assert i > j
            L_COORDINATE_MAP[(i, j)] = k
            L_nz_set_per_row[i].add(j)
            L_nz_set_per_col[j].add(i)

    # NOTE: we need to ensure these are in increasing order to access unused values.
    L_nz_per_row = [sorted(x) for x in L_nz_set_per_row]

    # NOTE: while the following can be in any order, we sort for consistency.
    L_nz_per_col = [sorted(x) for x in L_nz_set_per_col]

    ldlt_impl = ""

    L_filled = set()
    D_filled = set()

    for i in range(dim):
        for j in L_nz_per_row[i]:
            assert (i, j) in L_COORDINATE_MAP
            L_ij_idx = L_COORDINATE_MAP[(i, j)]
            line = f"    LT_data[{L_ij_idx}] = ("
            if (i, j) in N_COORDINATE_MAP:
                line += f"A_data[{N_COORDINATE_MAP[(i, j)]}]"
            else:
                line += "0.0"
            for k in sorted(L_nz_set_per_row[i].intersection(L_nz_set_per_row[j])):
                assert (i, k) in L_COORDINATE_MAP
                assert (j, k) in L_COORDINATE_MAP
                L_ik_idx = L_COORDINATE_MAP[(i, k)]
                L_jk_idx = L_COORDINATE_MAP[(j, k)]
                assert L_ik_idx in L_filled
                assert L_jk_idx in L_filled
                assert k in D_filled
                line += f" - (LT_data[{L_ik_idx}] * LT_data[{L_jk_idx}] * D_diag[{k}])"
            line += f") / D_diag[{j}];\n"
            ldlt_impl += line
            L_filled.add((L_ij_idx))

        # Update D_diag.
        line = f"    D_diag[{i}] = "
        if (i, i) in N_COORDINATE_MAP:
            line += f"A_data[{N_COORDINATE_MAP[(i, i)]}]"
        for j in L_nz_per_row[i]:
            assert (i, j) in L_COORDINATE_MAP
            L_ij_idx = L_COORDINATE_MAP[(i, j)]
            assert L_ij_idx in L_filled
            assert j in D_filled
            line += (
                f" - (LT_data[{L_ij_idx}] * LT_data[{L_ij_idx}] * D_diag[{j}])"
            )
        line += ";\n"
        ldlt_impl += line
        D_filled.add(i)

    solve_lower_unitriangular_impl = ""

    for i in range(dim):
        line = f"    x[{i}] = b[{i}]"
        for j in L_nz_per_row[i]:
            assert i > j
            assert (i, j) in L_COORDINATE_MAP
            L_ij_idx = L_COORDINATE_MAP[(i, j)]
            assert L_ij_idx in L_filled
            line += f" - LT_data[{L_ij_idx}] * x[{j}]"
        line += ";\n"
        solve_lower_unitriangular_impl += line

    solve_upper_unitriangular_impl = ""

    for i in range(dim - 1, -1, -1):
        line = f"    x[{i}] = b[{i}]"
        for j in L_nz_per_col[i]:
            assert j > i
            assert (j, i) in L_COORDINATE_MAP
            L_ji_idx = L_COORDINATE_MAP[(j, i)]
            assert L_ji_idx in L_filled
            line += f" - LT_data[{L_ji_idx}] * x[{j}]"
        line += ";\n"
        solve_upper_unitriangular_impl += line

    # NOTE:
    # 1. Mx = b iff (P_MAT M P_MAT.T) (P_MAT x) = (P_MAT b) iff (L D L.T) (P_MAT x) = (P_MAT b).
    # 2. First, set tmp2 = P_MAT b. Note tmp2[i] = (P_MAT b)[i] = sum_k P_MAT[i, k] b[k] = b[P[i]].
    # 3. Next, solve (L + I) tmp1 = tmp2.
    # 4. Next, do tmp1 /= D_diag.
    # 5. Next, solve (L.T + I) tmp2 = tmp1.
    # 6. Finally, solve P_MAT x = tmp2, i.e. set x = P_MAT.T tmp2. Note x[i] = (P_MAT.T tmp2)[i]
    #    = sum_k (P_MAT.T)[i, k] tmp2[k] = sum_k P_MAT[k, i] tmp2[k] = tmp2[PINV[i]].
    permute_b = ""
    for i in range(dim):
        permute_b += f"    tmp2[{i}] = b[{P[i]}];\n"

    permute_solution = ""
    for i in range(dim):
        permute_solution += f"    x[{P[i]}] = tmp2[{i}];\n"

    cpp_header_code = f"""#pragma once

namespace {namespace} {{

// Performs an L D L^T decomposition of the A matrix,
// where A_data is expected to represent np.triu(A) in CSC order.
// NOTE: LT_data and D_diag should have sizes L_nnz={L_nnz} and dim={dim} respectively.
void ldlt_factor(const double* A_data, double* LT_data, double* D_diag);

// Solves A * x = b, given a pre-computed L D L^T factorization of A.
// LT_data and D_diag can be computed via the ldlt_factor method defined above.
void ldlt_solve(const double* LT_data, const double* D_diag, const double* b, double* x);

}}  // namespace {namespace}\n"""

    cpp_impl_code = f"""#include "{header_name}.hpp"

#include <array>

namespace {namespace} {{

void ldlt_factor(const double* A_data, double* LT_data, double* D_diag) {{
{ldlt_impl}}}

namespace {{
void solve_lower_unitriangular(const double* LT_data, const double* b, double* x) {{
{solve_lower_unitriangular_impl}}}

void solve_upper_unitriangular(const double* LT_data, const double* b, double* x) {{
{solve_upper_unitriangular_impl}}}
}} // namespace

void ldlt_solve(const double* LT_data, const double* D_diag, const double* b, double* x) {{
    std::array<double, {dim}> tmp1;
    std::array<double, {dim}> tmp2;
{permute_b}
    solve_lower_unitriangular(LT_data, tmp2.data(), tmp1.data());
    for (std::size_t i = 0; i < {dim}; ++i) {{
        tmp1[i] /= D_diag[i];
    }}
    solve_upper_unitriangular(LT_data, tmp1.data(), tmp2.data());
{permute_solution}}}

}} // namespace {namespace}\n"""

    return cpp_header_code, cpp_impl_code
