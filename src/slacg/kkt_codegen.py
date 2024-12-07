import numpy as np
import scipy as sp

from src.slacg.internal.common import build_sparse_L


# Generates code for solving linear systems of the form Kx = b, where
# K = [[ H  C.T  G.T]
#      [ C  -rI   0 ]
#      [ G   0   -S ]].
# H is expected to be positive definite and S is expected to be a diagonal of positive numbers.
# r is also expected to be positive.
# The codegen interface is chosen so that factorizations can be re-used for successive solves
# sharing the same LHS matrix.


# NOTE:
# 1. Given a permutation P, we define the associated permutation matrix
#    P_MAT as P_MAT[i, j] = 0 iff j = p[i]. For example, if P = (2, 0, 1),
#    P_MAT = [[0, 0, 1],
#             [1, 0, 0],
#             [0, 1, 0]].
# 2. The user should pass a permutation P for which (P_MAT @ K @ P_MAT.T)
#    has as little fill-in as possible in its L D L^T decomposition.
#    Typically, an AMD ordering is pre-computed for the sparsity pattern.


def kkt_codegen(H, C, G, P, namespace, header_name):
    assert len(H.shape) == 2
    assert len(C.shape) == 2
    assert len(G.shape) == 2
    assert (H == H.T).all()
    x_dim = H.shape[0]
    y_dim = C.shape[0]
    s_dim = G.shape[0]
    dim = x_dim + y_dim + s_dim
    I_y = np.eye(y_dim)
    Zsy = np.zeros([s_dim, y_dim])
    Zys = Zsy.T
    S = np.eye(s_dim)
    # NOTE: in the code-definition of K, only the sparsity structure matters;
    #       it's irrelevant if certain blocks are off by a scalar multiplication.
    K = np.block([[H, C.T, G.T],
                  [C, I_y, Zys],
                  [G, Zsy,  S ]])

    SPARSE_L = build_sparse_L(M=K, P=P)

    L_nnz = SPARSE_L.nnz

    P_MAT = np.zeros_like(K)
    P_MAT[np.arange(dim), P] = 1.0

    N = P_MAT @ K @ P_MAT.T
    SPARSE_LOWER_N = sp.sparse.csc_matrix(np.tril(N))

    # NOTE:
    # 1. P_MAT[i, j] = 0 iff j = P[i]
    # 2. N[i, j] = (P_MAT K P_MAT.T)[i, j] = sum_k (P_MAT K)[i, k] (P_MAT.T)[k, j]
    #            = sum_k (P_MAT K)[i, k] P_MAT[j, k] = (P_MAT K)[i, P[j]]
    #            = sum_k P_MAT[i, k] K[k, P[j]] = K[P[i], P[j]]

    PINV = np.zeros_like(P)
    PINV[P] = np.arange(dim)

    SPARSE_UPPER_H = sp.sparse.csc_matrix(np.triu(H))

    # H_COORDINATE_MAP maps indices (i, j) of H (where i <= j) to the corresponding
    # data coordinate of SPARSE_UPPER_H.
    H_COORDINATE_MAP = {}
    for j in range(x_dim):
        for k in range(
            SPARSE_UPPER_H.indptr[j],
            SPARSE_UPPER_H.indptr[j + 1],
        ):
            i = int(SPARSE_UPPER_H.indices[k])
            assert i <= j
            H_COORDINATE_MAP[(i, j)] = k

    SPARSE_C = sp.sparse.csc_matrix(C)

    # C_COORDINATE_MAP maps indices (i, j) of C to the corresponding
    # data coordinate of SPARSE_C.
    C_COORDINATE_MAP = {}
    for j in range(x_dim):
        for k in range(
            SPARSE_C.indptr[j],
            SPARSE_C.indptr[j + 1],
        ):
            i = int(SPARSE_C.indices[k])
            C_COORDINATE_MAP[(i, j)] = k

    SPARSE_G = sp.sparse.csc_matrix(G)

    # G_COORDINATE_MAP maps indices (i, j) of G to the corresponding
    # data coordinate of SPARSE_G.
    G_COORDINATE_MAP = {}
    for j in range(x_dim):
        for k in range(
            SPARSE_G.indptr[j],
            SPARSE_G.indptr[j + 1],
        ):
            i = int(SPARSE_G.indices[k])
            G_COORDINATE_MAP[(i, j)] = k

    SPARSE_LOWER_K = sp.sparse.csc_matrix(np.tril(K))

    # K_COORDINATE_MAP maps indices (i, j) of K (where i >= j)
    # to code accessing the appropriate input value.
    K_COORDINATE_MAP = {}
    for j in range(dim):
        for k in range(
            SPARSE_LOWER_K.indptr[j],
            SPARSE_LOWER_K.indptr[j + 1],
        ):
            i = int(SPARSE_LOWER_K.indices[k])
            assert i >= j
            code = ""
            if i < x_dim:
                assert (j, i) in H_COORDINATE_MAP
                code = f"H_data[{H_COORDINATE_MAP[(j, i)]}]"
            elif i < x_dim + y_dim:
                if j < x_dim:
                    C_i = i - x_dim
                    C_j = j
                    assert (C_i, C_j) in C_COORDINATE_MAP
                    code = f"C_data[{C_COORDINATE_MAP[(C_i, C_j)]}]"
                else:
                    assert j < x_dim + y_dim
                    assert i == j
                    code = "(-r)"
            else:
                assert i < x_dim + y_dim + s_dim
                if j < x_dim:
                    G_i = i - x_dim - y_dim
                    G_j = j
                    assert (G_i, G_j) in G_COORDINATE_MAP
                    code = f"G_data[{G_COORDINATE_MAP[(G_i, G_j)]}]"
                else:
                    if i != j:
                        breakpoint()
                    assert i == j
                    s_i = i - x_dim - y_dim
                    code = f"(-s[{s_i}])"
            K_COORDINATE_MAP[(i, j)] = code

    # N_COORDINATE_MAP maps indices (i, j) of N (where i >= j)
    # to code accessing the appropriate input value.
    N_COORDINATE_MAP = {}
    for j in range(dim):
        for k in range(
            SPARSE_LOWER_N.indptr[j],
            SPARSE_LOWER_N.indptr[j + 1],
        ):
            i = int(SPARSE_LOWER_N.indices[k])
            assert i >= j
            m = int(P[i])
            n = int(P[j])
            if m < n:
                m, n = n, m
            assert N[i, j] == K[m, n]
            assert (m, n) in K_COORDINATE_MAP
            N_COORDINATE_MAP[(i, j)] = K_COORDINATE_MAP[(m, n)]

    L_COORDINATE_MAP = {}
    L_nz_set_per_row = [set() for _ in range(dim)]
    L_nz_set_per_col = [set() for _ in range(dim)]
    for j in range(dim):
        for k in range(SPARSE_L.indptr[j], SPARSE_L.indptr[j + 1]):
            i = int(SPARSE_L.indices[k])
            assert i > j
            L_COORDINATE_MAP[(i, j)] = k
            L_nz_set_per_row[i].add(j)
            L_nz_set_per_col[j].add(i)

    # NOTE: we need to ensure these are in increasing order to access unused values.
    L_nz_per_row = [sorted(x) for x in L_nz_set_per_row]

    ldlt_impl = ""

    L_filled = set()
    D_filled = set()

    for i in range(dim):
        for j in L_nz_per_row[i]:
            assert (i, j) in L_COORDINATE_MAP
            assert i > j
            L_ij_idx = L_COORDINATE_MAP[(i, j)]
            line = f"L_data[{L_ij_idx}] = ("
            if (i, j) in N_COORDINATE_MAP:
                line += N_COORDINATE_MAP[(i, j)]
            else:
                line += "0.0"
            for k in range(j):
                if (i, k) not in L_COORDINATE_MAP or (j, k) not in L_COORDINATE_MAP:
                    continue
                L_ik_idx = L_COORDINATE_MAP[(i, k)]
                L_jk_idx = L_COORDINATE_MAP[(j, k)]
                assert L_ik_idx in L_filled
                assert L_jk_idx in L_filled
                assert k in D_filled
                line += f" - (L_data[{L_ik_idx}] * L_data[{L_jk_idx}] * D_diag[{k}])"
            line += f") / D_diag[{j}];\n"
            ldlt_impl += line
            L_filled.add((L_ij_idx))

        # Update D_diag.
        line = f"D_diag[{i}] = "
        if (i, i) in N_COORDINATE_MAP:
            line += N_COORDINATE_MAP[(i, i)]
        for j in L_nz_per_row[i]:
            assert (i, j) in L_COORDINATE_MAP
            L_ij_idx = L_COORDINATE_MAP[(i, j)]
            assert L_ij_idx in L_filled
            assert j in D_filled
            line += (
                f" - (L_data[{L_ij_idx}] * L_data[{L_ij_idx}] * D_diag[{j}])"
            )
        line += ";\n"
        ldlt_impl += line
        D_filled.add(i)

    solve_lower_unitriangular_impl = ""

    for i in range(dim):
        line = f"x[{i}] = b[{i}]"
        for j in L_nz_per_row[i]:
            assert i > j
            assert (i, j) in L_COORDINATE_MAP
            L_ij_idx = L_COORDINATE_MAP[(i, j)]
            assert L_ij_idx in L_filled
            line += f" - L_data[{L_ij_idx}] * x[{j}]"
        line += ";\n"
        solve_lower_unitriangular_impl += line

    solve_upper_unitriangular_impl = ""

    for i in range(dim - 1, -1, -1):
        line = f"x[{i}] = b[{i}]"
        for j in L_nz_set_per_col[i]:
            assert j > i
            assert (j, i) in L_COORDINATE_MAP
            L_ji_idx = L_COORDINATE_MAP[(j, i)]
            assert L_ji_idx in L_filled
            line += f" - L_data[{L_ji_idx}] * x[{j}]"
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
        permute_b += f"tmp2[{i}] = b[{P[i]}];\n"

    permute_solution = ""
    for i in range(dim):
        permute_solution += f"x[{P[i]}] = tmp2[{i}];\n"

    add_Kx_to_y_impl = ""
    for j in range(K.shape[1]):
        for k in range(SPARSE_LOWER_K.indptr[j], SPARSE_LOWER_K.indptr[j + 1]):
            i = SPARSE_LOWER_K.indices[k]
            assert (i, j) in K_COORDINATE_MAP
            add_Kx_to_y_impl += (
                f"y[{i}] += {K_COORDINATE_MAP[(i, j)]} * x[{j}];\n"
            )
            if i != j:
                add_Kx_to_y_impl += (
                    f"y[{j}] += {K_COORDINATE_MAP[(i, j)]} * x[{i}];\n"
                )

    cpp_header_code = f"""
    #pragma once

    namespace {namespace}
    {{
    // Performs an L D L^T decomposition of the matrix (P_MAT * K * P_MAT.T), where
    // K = [[ H  C.T  G.T]
    //      [ C  -rI   0 ]
    //      [ G   0   -S ]],
    // where A_data is expected to represent np.triu(A) in CSC order.
    // NOTE: L_data and D_diag should have sizes L_nnz={L_nnz} and dim={dim} respectively.
    void ldlt_factor(const double* H_data, const double* C_data, const double* G_data,
                     const double* s, const double r,
                     double* L_data, double* D_diag);

    // Solves K * x = b, given a pre-computed L D L^T factorization of (P_MAT * K * P_MAT.T).
    // L_data and D_diag can be computed via the ldlt method defined above.
    void ldlt_solve(const double* L_data, const double* D_diag, const double* b, double* x);

    // Adds K * x to y.
    void add_Kx_to_y(const double* H_data, const double* C_data, const double* G_data,
                     const double* s, const double r, const double* x, double* y);
    }}  // namespace {namespace}
    """

    cpp_impl_code = f"""#include "{header_name}.hpp"

    #include <array>

    namespace {namespace}
    {{
    namespace {{
    void solve_lower_unitriangular(const double* L_data, const double* b, double* x) {{
    {solve_lower_unitriangular_impl}
    }}

    void solve_upper_unitriangular(const double* L_data, const double* b, double* x) {{
    {solve_upper_unitriangular_impl}
    }}
    }}  // namespace

    void ldlt_factor(const double* H_data, const double* C_data, const double* G_data,
                     const double* s, const double r,
                     double* L_data, double* D_diag) {{
    {ldlt_impl}
    }}

    void ldlt_solve(const double* L_data, const double* D_diag, const double* b, double* x) {{
    std::array<double, {dim}> tmp1;
    std::array<double, {dim}> tmp2;
    {permute_b}
    solve_lower_unitriangular(L_data, tmp2.data(), tmp1.data());
    for (std::size_t i = 0; i < {dim}; ++i) {{
        tmp1[i] /= D_diag[i];
    }}
    solve_upper_unitriangular(L_data, tmp1.data(), tmp2.data());
    {permute_solution}
    }}

    void add_Kx_to_y(const double* H_data, const double* C_data, const double* G_data,
                     const double* s, const double r, const double* x, double* y) {{
    {add_Kx_to_y_impl}
    }}

    }} // namespace {namespace}
    """

    return cpp_header_code, cpp_impl_code
