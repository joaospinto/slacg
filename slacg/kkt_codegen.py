import scipy as sp
import numpy as np

from slacg.internal.common import RESTRICT_MACRO, build_sparse_LT


# This file provides utilities for generating efficient code for solving
# Newton-KKT linear systems of the form Kx = k, where
# K = [[ H + r1 I_x     C.T         G.T     ]
#      [     C        -diag(r2)      0      ]
#      [     G           0      -W - diag(r3)]];
# the following properties are expected to hold:
# 1. (H + r1 I_x) is symmetric and positive definite;
# 2. W is diagonal and positive definite;
# 3. r1 is a non-negative regularization parameter;
# 4. r2 and r3 are vectors of non-negative regularization parameters.
# For performance (i.e. reducing fill-in), the user should also pass
# a permutation P so that an L D L^T decomposition of P_MAT @ K @ P_MAT.T
# is performed (instead of directly on K). Note:
# 1. Given a permutation P, we define the associated permutation matrix
#    P_MAT as P_MAT[i, j] = 0 iff j = p[i]. For example, if P = (2, 0, 1),
#    P_MAT = [[0, 0, 1],
#             [1, 0, 0],
#             [0, 1, 0]].
# 2. The user should pass a permutation P for which (P_MAT @ K @ P_MAT.T)
#    has as little fill-in as possible in its L D L^T decomposition.
#    Typically, an AMD ordering is pre-computed for the sparsity pattern.


def kkt_codegen(H, C, G, P, namespace, header_name):
    assert sp.sparse.issparse(H)
    assert sp.sparse.issparse(C)
    assert sp.sparse.issparse(G)
    assert len(H.shape) == 2
    assert len(C.shape) == 2
    assert len(G.shape) == 2
    H = H.tocsc(copy=True)
    C = C.tocsc(copy=True)
    G = G.tocsc(copy=True)
    H.eliminate_zeros()
    C.eliminate_zeros()
    G.eliminate_zeros()
    assert (H != H.T).nnz == 0
    P = np.asarray(P, dtype=int)
    x_dim = H.shape[0]
    y_dim = C.shape[0]
    z_dim = G.shape[0]
    dim = x_dim + y_dim + z_dim
    I_x = sp.sparse.eye(x_dim, format="csc")
    I_y = sp.sparse.eye(y_dim, format="csc")
    I_z = sp.sparse.eye(z_dim, format="csc")
    Zsy = sp.sparse.csc_matrix((z_dim, y_dim))
    Zys = Zsy.T
    H = abs(H) + I_x
    # NOTE: only the sparsity patterns matter here.
    K = sp.sparse.bmat(
        [[H, C.T, G.T], [C, I_y, Zys], [G, Zsy, I_z]], format="csc"
    )

    SPARSE_LT = build_sparse_LT(M=K, P=P)

    L_nnz = SPARSE_LT.nnz

    N = K.tocsr()[P, :][:, P].tocsc()
    SPARSE_LOWER_N = sp.sparse.tril(N, format="csc")

    # NOTE:
    # 1. P_MAT[i, j] = 0 iff j = P[i]
    # 2. N[i, j] = (P_MAT K P_MAT.T)[i, j] = K[P[i], P[j]]

    PINV = np.zeros_like(P)
    PINV[P] = np.arange(dim)

    SPARSE_UPPER_H = sp.sparse.triu(H, format="csc")

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

    SPARSE_C = C.tocsc()

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

    SPARSE_G = G.tocsc()

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

    SPARSE_LOWER_K = sp.sparse.tril(K, format="csc")

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
                if i == j:
                    code = f"(H_data[{H_COORDINATE_MAP[(j, i)]}] + r1)"
                else:
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
                    y_i = i - x_dim
                    code = f"(-r2[{y_i}])"
            else:
                assert i < x_dim + y_dim + z_dim
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
                    code = f"(-w[{s_i}] - r3[{s_i}])"
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

    # L_COORDINATE_MAP maps indices (i, j) of L (where i >= j)
    # to the corresponding data coordinate of SPARSE_LT.
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

    ldlt_impl = (
        "    int positive_count = 0;\n"
        "    int negative_count = 0;\n"
        "    double D_i;\n"
    )

    if y_dim == 0:
        ldlt_impl += "    (void) C_data;\n    (void) r2;\n"
    if z_dim == 0:
        ldlt_impl += "    (void) G_data;\n    (void) r3;\n"

    LT_filled = set()
    D_filled = set()

    for i in range(dim):
        for j in L_nz_per_row[i]:
            assert (i, j) in L_COORDINATE_MAP
            assert i > j
            L_ij_idx = L_COORDINATE_MAP[(i, j)]
            line = f"    LT_data[{L_ij_idx}] = "
            if (i, j) in N_COORDINATE_MAP:
                line += N_COORDINATE_MAP[(i, j)]
            for k in sorted(L_nz_set_per_row[i].intersection(L_nz_set_per_row[j])):
                assert (i, k) in L_COORDINATE_MAP
                assert (j, k) in L_COORDINATE_MAP
                L_ik_idx = L_COORDINATE_MAP[(i, k)]
                L_jk_idx = L_COORDINATE_MAP[(j, k)]
                assert L_ik_idx in LT_filled
                assert L_jk_idx in LT_filled
                assert k in D_filled
                line += f" - (LT_data[{L_ik_idx}] * LT_data[{L_jk_idx}])"
            line += f";\n"
            ldlt_impl += line
            LT_filled.add((L_ij_idx))

        # Update diagonal and finalize column of LT.
        line = "    D_i = "
        if (i, i) in N_COORDINATE_MAP:
            line += f"{N_COORDINATE_MAP[(i, i)]};\n"
        else:
            line += f"0.0;\n"
        ldlt_impl += line
        D_filled.add(i)
        for j in L_nz_per_row[i]:
            assert (i, j) in L_COORDINATE_MAP
            L_ij_idx = L_COORDINATE_MAP[(i, j)]
            assert L_ij_idx in LT_filled
            assert j in D_filled
            line = (
                f"    D_i -= LT_data[{L_ij_idx}] * "
                f"(LT_data[{L_ij_idx}] * D_inv[{j}]);\n"
            )
            ldlt_impl += line
            line = f"    LT_data[{L_ij_idx}] *= D_inv[{j}];\n"
            ldlt_impl += line
        ldlt_impl += (
            "    if (D_i > 0.0) {\n"
            "        ++positive_count;\n"
            "    } else if (D_i < 0.0) {\n"
            "        ++negative_count;\n"
            "    } else {\n"
            "        return false;\n"
            "    }\n"
        )
        ldlt_impl += f"    D_inv[{i}] = 1.0 / D_i;\n"

    ldlt_impl += (
        f"    return positive_count == {x_dim} && "
        f"negative_count == {y_dim + z_dim};\n"
    )

    solve_lower_unitriangular_impl = ""

    for i in range(dim):
        line = f"    x[{i}] = b[{i}]"
        for j in L_nz_per_row[i]:
            assert i > j
            assert (i, j) in L_COORDINATE_MAP
            L_ij_idx = L_COORDINATE_MAP[(i, j)]
            assert L_ij_idx in LT_filled
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
            assert L_ji_idx in LT_filled
            line += f" - LT_data[{L_ji_idx}] * x[{j}]"
        line += ";\n"
        solve_upper_unitriangular_impl += line

    # NOTE:
    # 1. Mx = b iff (P_MAT M P_MAT.T) (P_MAT x) = (P_MAT b) iff (L D L.T) (P_MAT x) = (P_MAT b).
    # 2. First, set tmp2 = P_MAT b. Note tmp2[i] = (P_MAT b)[i] = sum_k P_MAT[i, k] b[k] = b[P[i]].
    # 3. Next, solve (L + I) tmp1 = tmp2.
    # 4. Next, do tmp1 *= D_inv.
    # 5. Next, solve (L.T + I) tmp2 = tmp1.
    # 6. Finally, solve P_MAT x = tmp2, i.e. set x = P_MAT.T tmp2. Note x[i] = (P_MAT.T tmp2)[i]
    #    = sum_k (P_MAT.T)[i, k] tmp2[k] = sum_k P_MAT[k, i] tmp2[k] = tmp2[PINV[i]].
    permute_b = ""
    for i in range(dim):
        permute_b += f"    tmp2[{i}] = b[{P[i]}];\n"

    permute_solution = ""
    for i in range(dim):
        permute_solution += f"    x[{P[i]}] = tmp2[{i}];\n"

    scale_diagonal = ""
    for i in range(dim):
        scale_diagonal += f"    tmp1[{i}] *= D_inv[{i}];\n"

    add_upper_symmetric_Hx_to_y_impl = ""

    if SPARSE_UPPER_H.nnz == 0:
        add_upper_symmetric_Hx_to_y_impl += "    (void) H_data;\n    (void) x;\n    (void) y;\n"

    for j in range(H.shape[1]):
        for k in range(SPARSE_UPPER_H.indptr[j], SPARSE_UPPER_H.indptr[j + 1]):
            i = SPARSE_UPPER_H.indices[k]
            add_upper_symmetric_Hx_to_y_impl += f"    y[{i}] += H_data[{k}] * x[{j}];\n"
            if i != j:
                add_upper_symmetric_Hx_to_y_impl += f"    y[{j}] += H_data[{k}] * x[{i}];\n"

    add_CTx_to_y_impl = ""
    add_Cx_to_y_impl = ""
    add_CTx_and_Cx_to_y_impl = ""

    if SPARSE_C.nnz == 0:
        add_CTx_to_y_impl += "    (void) C_data;\n    (void) x;\n    (void) y;\n"
        add_Cx_to_y_impl += "    (void) C_data;\n    (void) x;\n    (void) y;\n"

    for j in range(C.shape[1]):
        col_start = SPARSE_C.indptr[j]
        col_end = SPARSE_C.indptr[j + 1]
        if col_start != col_end:
            add_CTx_and_Cx_to_y_impl += (
                f"    double y_x_C_{j} = y_x[{j}];\n"
                f"    const double x_x_C_{j} = x_x[{j}];\n"
            )
        for k in range(SPARSE_C.indptr[j], SPARSE_C.indptr[j + 1]):
            i = SPARSE_C.indices[k]
            add_CTx_to_y_impl += f"    y[{j}] += C_data[{k}] * x[{i}];\n"
            add_Cx_to_y_impl += f"    y[{i}] += C_data[{k}] * x[{j}];\n"
            add_CTx_and_Cx_to_y_impl += (
                f"    y_x_C_{j} += C_data[{k}] * x_y[{i}];\n"
                f"    y_y[{i}] += C_data[{k}] * x_x_C_{j};\n"
            )
        if col_start != col_end:
            add_CTx_and_Cx_to_y_impl += f"    y_x[{j}] = y_x_C_{j};\n"

    add_GTx_to_y_impl = ""
    add_Gx_to_y_impl = ""
    add_GTx_and_Gx_to_y_impl = ""

    if SPARSE_G.nnz == 0:
        add_GTx_to_y_impl += "    (void) G_data;\n    (void) x;\n    (void) y;\n"
        add_Gx_to_y_impl += "    (void) G_data;\n    (void) x;\n    (void) y;\n"

    for j in range(G.shape[1]):
        col_start = SPARSE_G.indptr[j]
        col_end = SPARSE_G.indptr[j + 1]
        if col_start != col_end:
            add_GTx_and_Gx_to_y_impl += (
                f"    double y_x_G_{j} = y_x[{j}];\n"
                f"    const double x_x_G_{j} = x_x[{j}];\n"
            )
        for k in range(SPARSE_G.indptr[j], SPARSE_G.indptr[j + 1]):
            i = SPARSE_G.indices[k]
            add_GTx_to_y_impl += f"    y[{j}] += G_data[{k}] * x[{i}];\n"
            add_Gx_to_y_impl += f"    y[{i}] += G_data[{k}] * x[{j}];\n"
            add_GTx_and_Gx_to_y_impl += (
                f"    y_x_G_{j} += G_data[{k}] * x_z[{i}];\n"
                f"    y_z[{i}] += G_data[{k}] * x_x_G_{j};\n"
            )
        if col_start != col_end:
            add_GTx_and_Gx_to_y_impl += f"    y_x[{j}] = y_x_G_{j};\n"

    cpp_header_code = f"""#pragma once

{RESTRICT_MACRO}

namespace {namespace} {{

constexpr int L_nnz = {L_nnz};

constexpr int dim = {dim};
constexpr int x_dim = {x_dim};
constexpr int y_dim = {y_dim};
constexpr int z_dim = {z_dim};

// Performs an L D L^T decomposition of the matrix (P_MAT * K * P_MAT.T), where
// K = [[ H + r1 I   C.T     G.T    ]
//      [    C     -diag(r2)    0       ]
//      [    G         0    -W - diag(r3)]],
// where:
// 1. H_data is expected to represent np.triu(H) in CSC order.
// 2. C_data and G_data are expected to represent C and G, respectively, in CSC order.
// 3. W is a diagonal matrix, represented by the vector of its diagonal elements, w.
// Returns true iff the computed factorization has the expected KKT inertia:
// x_dim positive pivots and y_dim + z_dim negative pivots.
// NOTE: LT_data and D_inv should have sizes L_nnz={L_nnz} and dim={dim} respectively.
bool ldlt_factor(const double *SLACG_RESTRICT H_data,
                 const double *SLACG_RESTRICT C_data,
                 const double *SLACG_RESTRICT G_data,
                 const double *SLACG_RESTRICT w, const double r1,
                 const double *SLACG_RESTRICT r2,
                 const double *SLACG_RESTRICT r3,
                 double *SLACG_RESTRICT LT_data,
                 double *SLACG_RESTRICT D_inv);

// Solves K * x = b, given a pre-computed L D L^T factorization of (P_MAT * K * P_MAT.T).
// LT_data and D_inv can be computed via the ldlt_factor method defined above.
void ldlt_solve(const double *SLACG_RESTRICT LT_data,
                const double *SLACG_RESTRICT D_inv,
                const double *SLACG_RESTRICT b,
                double *SLACG_RESTRICT x);

// Adds H @ x to y.
void add_upper_symmetric_Hx_to_y(const double *SLACG_RESTRICT H_data,
                                 const double *SLACG_RESTRICT x,
                                 double *SLACG_RESTRICT y);

// Adds C.T @ x to y.
void add_CTx_to_y(const double *SLACG_RESTRICT C_data,
                  const double *SLACG_RESTRICT x,
                  double *SLACG_RESTRICT y);

// Adds C @ x to y.
void add_Cx_to_y(const double *SLACG_RESTRICT C_data,
                 const double *SLACG_RESTRICT x,
                 double *SLACG_RESTRICT y);

// Adds G.T @ x to y.
void add_GTx_to_y(const double *SLACG_RESTRICT G_data,
                  const double *SLACG_RESTRICT x,
                  double *SLACG_RESTRICT y);

// Adds G @ x to y.
void add_Gx_to_y(const double *SLACG_RESTRICT G_data,
                 const double *SLACG_RESTRICT x,
                 double *SLACG_RESTRICT y);

// Adds K * x to y, where
// K = [[ H + r1 I   C.T     G.T    ]
//      [    C     -diag(r2)    0       ]
//      [    G         0    -W - diag(r3)]].
void add_Kx_to_y(const double *SLACG_RESTRICT H_data,
                 const double *SLACG_RESTRICT C_data,
                 const double *SLACG_RESTRICT G_data,
                 const double *SLACG_RESTRICT w, const double r1,
                 const double *SLACG_RESTRICT r2,
                 const double *SLACG_RESTRICT r3,
                 const double *SLACG_RESTRICT x_x,
                 const double *SLACG_RESTRICT x_y,
                 const double *SLACG_RESTRICT x_z,
                 double *SLACG_RESTRICT y_x,
                 double *SLACG_RESTRICT y_y,
                 double *SLACG_RESTRICT y_z);

}}  // namespace {namespace}\n"""

    cpp_impl_code = f"""#include "{header_name}.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace {namespace} {{

namespace {{
void solve_lower_unitriangular(const double *SLACG_RESTRICT LT_data,
                               const double *SLACG_RESTRICT b,
                               double *SLACG_RESTRICT x) {{
{solve_lower_unitriangular_impl}}}

void solve_upper_unitriangular(const double *SLACG_RESTRICT LT_data,
                               const double *SLACG_RESTRICT b,
                               double *SLACG_RESTRICT x) {{
{solve_upper_unitriangular_impl}}}
}}  // namespace

bool ldlt_factor(const double *SLACG_RESTRICT H_data,
                 const double *SLACG_RESTRICT C_data,
                 const double *SLACG_RESTRICT G_data,
                 const double *SLACG_RESTRICT w, const double r1,
                 const double *SLACG_RESTRICT r2,
                 const double *SLACG_RESTRICT r3,
                 double *SLACG_RESTRICT LT_data,
                 double *SLACG_RESTRICT D_inv) {{
{ldlt_impl}}}

void ldlt_solve(const double *SLACG_RESTRICT LT_data,
                const double *SLACG_RESTRICT D_inv,
                const double *SLACG_RESTRICT b,
                double *SLACG_RESTRICT x) {{
    std::array<double, {dim}> tmp1;
    std::array<double, {dim}> tmp2;
{permute_b}
    solve_lower_unitriangular(LT_data, tmp2.data(), tmp1.data());
{scale_diagonal}
    solve_upper_unitriangular(LT_data, tmp1.data(), tmp2.data());

{permute_solution}}}

void add_Kx_to_y(const double *SLACG_RESTRICT H_data,
                 const double *SLACG_RESTRICT C_data,
                 const double *SLACG_RESTRICT G_data,
                 const double *SLACG_RESTRICT w, const double r1,
                 const double *SLACG_RESTRICT r2,
                 const double *SLACG_RESTRICT r3,
                 const double *SLACG_RESTRICT x_x,
                 const double *SLACG_RESTRICT x_y,
                 const double *SLACG_RESTRICT x_z,
                 double *SLACG_RESTRICT y_x,
                 double *SLACG_RESTRICT y_y,
                 double *SLACG_RESTRICT y_z) {{
    add_upper_symmetric_Hx_to_y(H_data, x_x, y_x);

{add_CTx_and_Cx_to_y_impl}

{add_GTx_and_Gx_to_y_impl}

    for (int i = 0; i < {x_dim}; ++i) {{
        y_x[i] += r1 * x_x[i];
    }}

    for (int i = 0; i < {y_dim}; ++i) {{
      y_y[i] -= r2[i] * x_y[i];
    }}

    for (int i = 0; i < {z_dim}; ++i) {{
      y_z[i] -= (w[i] + r3[i]) * x_z[i];
    }}
}}

void add_upper_symmetric_Hx_to_y(const double *SLACG_RESTRICT H_data,
                                 const double *SLACG_RESTRICT x,
                                 double *SLACG_RESTRICT y) {{
{add_upper_symmetric_Hx_to_y_impl}
}}

void add_CTx_to_y(const double *SLACG_RESTRICT C_data,
                  const double *SLACG_RESTRICT x,
                  double *SLACG_RESTRICT y) {{
{add_CTx_to_y_impl}
}}

void add_Cx_to_y(const double *SLACG_RESTRICT C_data,
                 const double *SLACG_RESTRICT x,
                 double *SLACG_RESTRICT y) {{
{add_Cx_to_y_impl}
}}

void add_GTx_to_y(const double *SLACG_RESTRICT G_data,
                  const double *SLACG_RESTRICT x,
                  double *SLACG_RESTRICT y) {{
{add_GTx_to_y_impl}
}}

void add_Gx_to_y(const double *SLACG_RESTRICT G_data,
                 const double *SLACG_RESTRICT x,
                 double *SLACG_RESTRICT y) {{
{add_Gx_to_y_impl}
}}

}} // namespace {namespace}\n"""

    return cpp_header_code, cpp_impl_code
