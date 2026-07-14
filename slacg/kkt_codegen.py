import re
from dataclasses import dataclass
from operator import index
from pathlib import Path

import numpy as np
import scipy as sp

from slacg.internal.common import RESTRICT_MACRO, build_sparse_LT


@dataclass(frozen=True)
class GeneratedFile:
    name: str
    contents: str


def write_generated_files(output_directory, generated_files):
    output_directory = Path(output_directory)
    for generated_file in generated_files:
        (output_directory / generated_file.name).write_text(
            generated_file.contents, encoding="utf-8"
        )


def _implementation_code(header_name, namespace, system_headers, body):
    includes = "".join(f"#include <{header}>\n" for header in system_headers)
    if includes:
        includes = f"\n{includes}"
    return f"""#include "{header_name}.hpp"
{includes}
namespace {namespace} {{

{body}

}}  // namespace {namespace}
"""


def _partition_contiguous_by_size(items, num_partitions):
    cumulative_sizes = [0]
    for item in items:
        cumulative_sizes.append(cumulative_sizes[-1] + len(item))

    cut_indices = [0]
    for partition in range(1, num_partitions):
        minimum = cut_indices[-1] + 1
        maximum = len(items) - (num_partitions - partition)
        target = cumulative_sizes[-1] * partition / num_partitions
        cut_indices.append(
            min(
                range(minimum, maximum + 1),
                key=lambda item_index: abs(cumulative_sizes[item_index] - target),
            )
        )
    cut_indices.append(len(items))
    return tuple(
        items[cut_indices[i] : cut_indices[i + 1]] for i in range(num_partitions)
    )


def _validate_num_chunks(value, dimension, parameter_name):
    try:
        value = index(value)
    except TypeError as error:
        raise TypeError(f"{parameter_name} must be an integer") from error
    if value < 1 or value > dimension:
        raise ValueError(f"{parameter_name} must be between 1 and {dimension}")
    return value


def _split_implementation_lines(implementations, max_lines):
    blocks = []
    for implementation in implementations:
        lines = implementation.splitlines(keepends=True)
        blocks.extend(
            "".join(lines[start : start + max_lines])
            for start in range(0, len(lines), max_lines)
        )
    return tuple(blocks)


def _partition_implementations(implementations, num_partitions):
    implementations = tuple(implementations)
    if len(implementations) < num_partitions:
        implementations += ("",) * (num_partitions - len(implementations))
    return _partition_contiguous_by_size(implementations, num_partitions)


def _fused_product_implementations(
    matrix,
    matrix_name,
    other_x_name,
    other_y_name,
    max_entries_per_block=None,
):
    implementations = []
    for j in range(matrix.shape[1]):
        col_start = matrix.indptr[j]
        col_end = matrix.indptr[j + 1]
        if col_start == col_end:
            continue
        block_size = max_entries_per_block or (col_end - col_start)
        for block_start in range(col_start, col_end, block_size):
            block_end = min(block_start + block_size, col_end)
            needs_scope = col_end - col_start > block_size
            lines = []
            if needs_scope:
                lines.append("    {\n")
            lines.extend(
                [
                    f"        double y_x_{matrix_name}_{j} = y_x[{j}];\n",
                    f"        const double x_x_{matrix_name}_{j} = x_x[{j}];\n",
                ]
            )
            for k in range(block_start, block_end):
                i = matrix.indices[k]
                lines.extend(
                    [
                        f"        y_x_{matrix_name}_{j} += "
                        f"{matrix_name}_data[{k}] * {other_x_name}[{i}];\n",
                        f"        {other_y_name}[{i}] += {matrix_name}_data[{k}] "
                        f"* x_x_{matrix_name}_{j};\n",
                    ]
                )
            lines.append(f"        y_x[{j}] = y_x_{matrix_name}_{j};\n")
            if needs_scope:
                lines.append("    }\n")
            implementations.append("".join(lines))
    return tuple(implementations)


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


def kkt_codegen(
    H,
    C,
    G,
    P,
    namespace,
    header_name,
    num_factor_chunks=1,
    num_solve_chunks=1,
    num_product_chunks=1,
    bordered_x_indices=(),
):
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
    bordered_x_indices = tuple(index(i) for i in bordered_x_indices)
    if len(set(bordered_x_indices)) != len(bordered_x_indices):
        raise ValueError("bordered_x_indices must not contain duplicates")
    if any(i < 0 or i >= x_dim for i in bordered_x_indices):
        raise ValueError("bordered_x_indices must contain valid x indices")
    border_dim = len(bordered_x_indices)
    core_indices = tuple(i for i in range(dim) if i not in bordered_x_indices)
    core_dim = len(core_indices)
    if core_dim == 0:
        raise ValueError("the bordered system must have a nonempty core")
    full_to_core = {full_index: i for i, full_index in enumerate(core_indices)}
    num_factor_chunks = _validate_num_chunks(
        num_factor_chunks, core_dim, "num_factor_chunks"
    )
    num_solve_chunks = _validate_num_chunks(
        num_solve_chunks, core_dim, "num_solve_chunks"
    )
    num_product_chunks = _validate_num_chunks(
        num_product_chunks, x_dim, "num_product_chunks"
    )
    I_x = sp.sparse.eye(x_dim, format="csc")
    I_y = sp.sparse.eye(y_dim, format="csc")
    I_z = sp.sparse.eye(z_dim, format="csc")
    Zsy = sp.sparse.csc_matrix((z_dim, y_dim))
    Zys = Zsy.T
    H = abs(H) + I_x
    # NOTE: only the sparsity patterns matter here.
    K = sp.sparse.bmat([[H, C.T, G.T], [C, I_y, Zys], [G, Zsy, I_z]], format="csc")

    if P.shape != (dim,) or set(P.tolist()) != set(range(dim)):
        raise ValueError("P must be a permutation of the KKT indices")
    permuted_core_indices = tuple(int(i) for i in P if i in full_to_core)
    core_permutation = np.asarray(
        [full_to_core[i] for i in permuted_core_indices], dtype=int
    )
    CORE_K = K.tocsr()[core_indices, :][:, core_indices].tocsc()
    SPARSE_LT = build_sparse_LT(M=CORE_K, P=core_permutation)

    L_nnz = SPARSE_LT.nnz

    N = K.tocsr()[permuted_core_indices, :][:, permuted_core_indices].tocsc()
    SPARSE_LOWER_N = sp.sparse.tril(N, format="csc")

    # NOTE:
    # 1. P_MAT[i, j] = 0 iff j = P[i]
    # 2. N[i, j] = (P_MAT K P_MAT.T)[i, j] = K[P[i], P[j]]

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
                    assert i == j
                    s_i = i - x_dim - y_dim
                    code = f"(-w[{s_i}] - r3[{s_i}])"
            K_COORDINATE_MAP[(i, j)] = code

    # N_COORDINATE_MAP maps indices (i, j) of N (where i >= j)
    # to code accessing the appropriate input value.
    N_COORDINATE_MAP = {}
    for j in range(core_dim):
        for k in range(
            SPARSE_LOWER_N.indptr[j],
            SPARSE_LOWER_N.indptr[j + 1],
        ):
            i = int(SPARSE_LOWER_N.indices[k])
            assert i >= j
            m = permuted_core_indices[i]
            n = permuted_core_indices[j]
            if m < n:
                m, n = n, m
            assert N[i, j] == K[m, n]
            assert (m, n) in K_COORDINATE_MAP
            N_COORDINATE_MAP[(i, j)] = K_COORDINATE_MAP[(m, n)]

    # L_COORDINATE_MAP maps indices (i, j) of L (where i >= j)
    # to the corresponding data coordinate of SPARSE_LT.
    L_COORDINATE_MAP = {}
    L_nz_set_per_row = [set() for _ in range(core_dim)]
    L_nz_set_per_col = [set() for _ in range(core_dim)]
    for i in range(core_dim):
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

    ldlt_prefix = "    int positive_count = 0;\n    int negative_count = 0;\n"

    if y_dim == 0:
        ldlt_prefix += "    (void) C_data;\n    (void) r2;\n"
    if z_dim == 0:
        ldlt_prefix += "    (void) G_data;\n    (void) w;\n    (void) r3;\n"

    LT_filled = set()
    D_filled = set()
    ldlt_pivot_implementations = []
    ldlt_chunkable_implementations = []

    for i in range(core_dim):
        pivot_implementation = ""
        for j in L_nz_per_row[i]:
            assert (i, j) in L_COORDINATE_MAP
            assert i > j
            L_ij_idx = L_COORDINATE_MAP[(i, j)]
            line = f"    LT_data[{L_ij_idx}] = "
            initial_value = "0.0"
            if (i, j) in N_COORDINATE_MAP:
                initial_value = N_COORDINATE_MAP[(i, j)]
                line += initial_value
            subtraction_terms = []
            for k in sorted(L_nz_set_per_row[i].intersection(L_nz_set_per_row[j])):
                assert (i, k) in L_COORDINATE_MAP
                assert (j, k) in L_COORDINATE_MAP
                L_ik_idx = L_COORDINATE_MAP[(i, k)]
                L_jk_idx = L_COORDINATE_MAP[(j, k)]
                assert L_ik_idx in LT_filled
                assert L_jk_idx in LT_filled
                assert k in D_filled
                term = f"(LT_data[{L_ik_idx}] * LT_data[{L_jk_idx}])"
                subtraction_terms.append(term)
                line += f" - {term}"
            line += ";\n"
            pivot_implementation += line
            for block_start in range(0, max(1, len(subtraction_terms)), 128):
                block_terms = subtraction_terms[block_start : block_start + 128]
                block_initial_value = (
                    initial_value if block_start == 0 else f"LT_data[{L_ij_idx}]"
                )
                block_lines = [
                    "    {\n",
                    f"        double value = {block_initial_value};\n",
                ]
                block_lines.extend(
                    f"        value -= {term};\n" for term in block_terms
                )
                block_lines.extend(
                    [f"        LT_data[{L_ij_idx}] = value;\n", "    }\n"]
                )
                ldlt_chunkable_implementations.append("".join(block_lines))
            LT_filled.add((L_ij_idx))

        # Update diagonal and finalize column of LT.
        line = "    D_i = "
        if (i, i) in N_COORDINATE_MAP:
            line += f"{N_COORDINATE_MAP[(i, i)]};\n"
        else:
            line += "0.0;\n"
        pivot_implementation += line
        ldlt_chunkable_implementations.append(line)
        D_filled.add(i)
        for j in L_nz_per_row[i]:
            assert (i, j) in L_COORDINATE_MAP
            L_ij_idx = L_COORDINATE_MAP[(i, j)]
            assert L_ij_idx in LT_filled
            assert j in D_filled
            update_implementation = (
                f"    const double LT_{i}_{j} = LT_data[{L_ij_idx}];\n"
                f"    const double normalized_LT_{i}_{j} = LT_{i}_{j} * D_inv[{j}];\n"
                f"    D_i -= LT_{i}_{j} * normalized_LT_{i}_{j};\n"
                f"    LT_data[{L_ij_idx}] = normalized_LT_{i}_{j};\n"
            )
            pivot_implementation += update_implementation
            ldlt_chunkable_implementations.append(update_implementation)
        pivot_suffix = (
            "    if (!std::isfinite(D_i)) {\n"
            "        return FactorStatus::kNonFinitePivot;\n"
            "    }\n"
            "    if (D_i > 0.0) {\n"
            "        ++positive_count;\n"
            "    } else if (D_i < 0.0) {\n"
            "        ++negative_count;\n"
            "    } else {\n"
            "        return FactorStatus::kZeroPivot;\n"
            "    }\n"
        )
        pivot_suffix += f"    D_inv[{i}] = 1.0 / D_i;\n"
        pivot_implementation += pivot_suffix
        ldlt_chunkable_implementations.append(pivot_suffix)
        ldlt_pivot_implementations.append(pivot_implementation)

    ldlt_suffix = (
        "    if (positive_count != x_dim - border_dim || "
        "negative_count != expected_negative_inertia) {\n"
        "        return FactorStatus::kWrongInertia;\n"
        "    }\n"
    )

    solve_lower_unitriangular_implementations = []

    for i in range(core_dim):
        lines = ["    {\n", f"        double value = b[{i}];\n"]
        for j in L_nz_per_row[i]:
            assert i > j
            assert (i, j) in L_COORDINATE_MAP
            L_ij_idx = L_COORDINATE_MAP[(i, j)]
            assert L_ij_idx in LT_filled
            lines.append(f"        value -= LT_data[{L_ij_idx}] * x[{j}];\n")
        lines.extend([f"        x[{i}] = value;\n", "    }\n"])
        solve_lower_unitriangular_implementations.append("".join(lines))

    solve_upper_unitriangular_implementations = []

    for i in range(core_dim - 1, -1, -1):
        lines = ["    {\n", f"        double value = b[{i}];\n"]
        for j in L_nz_per_col[i]:
            assert j > i
            assert (j, i) in L_COORDINATE_MAP
            L_ji_idx = L_COORDINATE_MAP[(j, i)]
            assert L_ji_idx in LT_filled
            lines.append(f"        value -= LT_data[{L_ji_idx}] * x[{j}];\n")
        lines.extend([f"        x[{i}] = value;\n", "    }\n"])
        solve_upper_unitriangular_implementations.append("".join(lines))

    # NOTE:
    # 1. Mx = b iff (P_MAT M P_MAT.T) (P_MAT x) = (P_MAT b) iff (L D L.T) (P_MAT x) = (P_MAT b).
    # 2. First, set tmp2 = P_MAT b. Note tmp2[i] = (P_MAT b)[i] = sum_k P_MAT[i, k] b[k] = b[P[i]].
    # 3. Next, solve (L + I) tmp1 = tmp2.
    # 4. Next, do tmp1 *= D_inv.
    # 5. Next, solve (L.T + I) tmp2 = tmp1.
    # 6. Finally, solve P_MAT x = tmp2, i.e. set x = P_MAT.T tmp2. Note x[i] = (P_MAT.T tmp2)[i]
    #    = sum_k (P_MAT.T)[i, k] tmp2[k] = sum_k P_MAT[k, i] tmp2[k] = tmp2[PINV[i]].
    permute_b_implementations = [
        f"    tmp2[{i}] = b[{permuted_core_indices[i]}];\n" for i in range(core_dim)
    ]
    permute_solution_implementations = [
        f"    x[{permuted_core_indices[i]}] = tmp2[{i}];\n" for i in range(core_dim)
    ]
    scale_diagonal_implementations = [
        f"    tmp1[{i}] *= D_inv[{i}];\n" for i in range(core_dim)
    ]

    add_upper_symmetric_Hx_to_y_prefix = ""
    if SPARSE_UPPER_H.nnz == 0:
        add_upper_symmetric_Hx_to_y_prefix = (
            "    (void) H_data;\n    (void) x;\n    (void) y;\n"
        )
    add_upper_symmetric_Hx_to_y_implementations = []
    for j in range(H.shape[1]):
        implementation = ""
        for k in range(SPARSE_UPPER_H.indptr[j], SPARSE_UPPER_H.indptr[j + 1]):
            i = SPARSE_UPPER_H.indices[k]
            implementation += f"    y[{i}] += H_data[{k}] * x[{j}];\n"
            if i != j:
                implementation += f"    y[{j}] += H_data[{k}] * x[{i}];\n"
        add_upper_symmetric_Hx_to_y_implementations.append(implementation)

    add_CTx_to_y_prefix = ""
    add_Cx_to_y_prefix = ""
    add_CTx_and_Cx_to_y_prefix = ""
    if SPARSE_C.nnz == 0:
        add_CTx_to_y_prefix = "    (void) C_data;\n    (void) x;\n    (void) y;\n"
        add_Cx_to_y_prefix = "    (void) C_data;\n    (void) x;\n    (void) y;\n"
        add_CTx_and_Cx_to_y_prefix = (
            "    (void) C_data;\n    (void) x_x;\n    (void) x_y;\n"
            "    (void) y_x;\n    (void) y_y;\n"
        )
    add_CTx_to_y_implementations = []
    add_Cx_to_y_implementations = []
    for j in range(C.shape[1]):
        transpose_implementation = ""
        forward_implementation = ""
        for k in range(SPARSE_C.indptr[j], SPARSE_C.indptr[j + 1]):
            i = SPARSE_C.indices[k]
            transpose_implementation += f"    y[{j}] += C_data[{k}] * x[{i}];\n"
            forward_implementation += f"    y[{i}] += C_data[{k}] * x[{j}];\n"
        add_CTx_to_y_implementations.append(transpose_implementation)
        add_Cx_to_y_implementations.append(forward_implementation)
    add_CTx_and_Cx_to_y_implementations = _fused_product_implementations(
        SPARSE_C, "C", "x_y", "y_y"
    )

    add_GTx_to_y_prefix = ""
    add_Gx_to_y_prefix = ""
    add_GTx_and_Gx_to_y_prefix = ""
    if SPARSE_G.nnz == 0:
        add_GTx_to_y_prefix = "    (void) G_data;\n    (void) x;\n    (void) y;\n"
        add_Gx_to_y_prefix = "    (void) G_data;\n    (void) x;\n    (void) y;\n"
        add_GTx_and_Gx_to_y_prefix = (
            "    (void) G_data;\n    (void) x_x;\n    (void) x_z;\n"
            "    (void) y_x;\n    (void) y_z;\n"
        )
    add_GTx_to_y_implementations = []
    add_Gx_to_y_implementations = []
    for j in range(G.shape[1]):
        transpose_implementation = ""
        forward_implementation = ""
        for k in range(SPARSE_G.indptr[j], SPARSE_G.indptr[j + 1]):
            i = SPARSE_G.indices[k]
            transpose_implementation += f"    y[{j}] += G_data[{k}] * x[{i}];\n"
            forward_implementation += f"    y[{i}] += G_data[{k}] * x[{j}];\n"
        add_GTx_to_y_implementations.append(transpose_implementation)
        add_Gx_to_y_implementations.append(forward_implementation)
    add_GTx_and_Gx_to_y_implementations = _fused_product_implementations(
        SPARSE_G, "G", "x_z", "y_z"
    )

    add_upper_symmetric_Hx_to_y_impl = add_upper_symmetric_Hx_to_y_prefix + "".join(
        add_upper_symmetric_Hx_to_y_implementations
    )
    add_CTx_to_y_impl = add_CTx_to_y_prefix + "".join(add_CTx_to_y_implementations)
    add_Cx_to_y_impl = add_Cx_to_y_prefix + "".join(add_Cx_to_y_implementations)
    add_CTx_and_Cx_to_y_impl = add_CTx_and_Cx_to_y_prefix + "".join(
        add_CTx_and_Cx_to_y_implementations
    )
    add_GTx_to_y_impl = add_GTx_to_y_prefix + "".join(add_GTx_to_y_implementations)
    add_Gx_to_y_impl = add_Gx_to_y_prefix + "".join(add_Gx_to_y_implementations)
    add_GTx_and_Gx_to_y_impl = add_GTx_and_Gx_to_y_prefix + "".join(
        add_GTx_and_Gx_to_y_implementations
    )

    def kkt_value(row, col):
        if row < col:
            row, col = col, row
        return K_COORDINATE_MAP.get((row, col))

    border_columns = []
    for border_index in bordered_x_indices:
        border_columns.append(
            tuple(
                (core_index, value)
                for core_index in core_indices
                if (value := kkt_value(core_index, border_index)) is not None
            )
        )

    if border_dim == 0:
        border_factor_implementation = (
            "    (void) border_solution;\n"
            "    (void) border_factor;\n"
            "    return FactorStatus::kSuccess;\n"
        )
        border_solve_prefix = "    (void) border_solution;\n    (void) border_factor;\n"
        border_rhs_implementations = []
        border_solve_middle = ""
        border_correction_implementations = []
        border_solve_suffix = ""
    else:
        border_factor_lines = [
            f"    std::array<double, {dim}> border_rhs{{}};\n",
        ]
        for col, entries in enumerate(border_columns):
            border_factor_lines.append("    border_rhs.fill(0.0);\n")
            border_factor_lines.extend(
                f"    border_rhs[{core_index}] = {value};\n"
                for core_index, value in entries
            )
            border_factor_lines.append(
                "    internal::ldlt_solve_core(LT_data, D_inv, "
                f"border_rhs.data(), border_solution + {col * dim});\n"
            )

        for col, border_col in enumerate(bordered_x_indices):
            for row in range(col, border_dim):
                diagonal_value = kkt_value(bordered_x_indices[row], border_col)
                initial_value = diagonal_value or "0.0"
                border_factor_lines.extend(
                    ["    {\n", f"        double value = {initial_value};\n"]
                )
                border_factor_lines.extend(
                    f"        value -= {entry_value} * "
                    f"border_solution[{col * dim + core_index}];\n"
                    for core_index, entry_value in border_columns[row]
                )
                border_factor_lines.extend(
                    [
                        f"        border_factor[{row + col * border_dim}] = value;\n",
                        "    }\n",
                    ]
                )

        for col in range(border_dim):
            for row in range(col, border_dim):
                border_factor_lines.extend(
                    [
                        "    {\n",
                        f"        double value = border_factor[{row + col * border_dim}];\n",
                    ]
                )
                border_factor_lines.extend(
                    f"        value -= border_factor[{row + k * border_dim}] * "
                    f"border_factor[{col + k * border_dim}];\n"
                    for k in range(col)
                )
                if row == col:
                    border_factor_lines.extend(
                        [
                            "        if (!std::isfinite(value)) {\n",
                            "            return FactorStatus::kNonFinitePivot;\n",
                            "        }\n",
                            "        if (value < 0.0) {\n",
                            "            return FactorStatus::kWrongInertia;\n",
                            "        }\n",
                            "        if (value == 0.0) {\n",
                            "            return FactorStatus::kZeroPivot;\n",
                            "        }\n",
                            f"        border_factor[{row + col * border_dim}] = "
                            "std::sqrt(value);\n",
                        ]
                    )
                else:
                    border_factor_lines.append(
                        f"        border_factor[{row + col * border_dim}] = value / "
                        f"border_factor[{col + col * border_dim}];\n"
                    )
                border_factor_lines.append("    }\n")
        border_factor_lines.append("    return FactorStatus::kSuccess;\n")
        border_factor_implementation = "".join(border_factor_lines)

        border_solve_prefix_lines = [
            f"    std::array<double, {border_dim}> theta;\n",
        ]
        for col, border_index in enumerate(bordered_x_indices):
            border_solve_prefix_lines.append(f"    theta[{col}] = b[{border_index}];\n")
        border_rhs_implementations = [
            "".join(
                f"    theta[{col}] -= "
                f"border_solution[{col * dim + core_index}] * b[{core_index}];\n"
                for col in range(border_dim)
            )
            for core_index in core_indices
        ]
        border_solve_middle_lines = []
        for row in range(border_dim):
            border_solve_middle_lines.extend(
                ["    {\n", f"        double value = theta[{row}];\n"]
            )
            border_solve_middle_lines.extend(
                f"        value -= border_factor[{row + col * border_dim}] * "
                f"theta[{col}];\n"
                for col in range(row)
            )
            border_solve_middle_lines.extend(
                [
                    f"        theta[{row}] = value / "
                    f"border_factor[{row + row * border_dim}];\n",
                    "    }\n",
                ]
            )
        for row in range(border_dim - 1, -1, -1):
            border_solve_middle_lines.extend(
                ["    {\n", f"        double value = theta[{row}];\n"]
            )
            border_solve_middle_lines.extend(
                f"        value -= border_factor[{col + row * border_dim}] * "
                f"theta[{col}];\n"
                for col in range(row + 1, border_dim)
            )
            border_solve_middle_lines.extend(
                [
                    f"        theta[{row}] = value / "
                    f"border_factor[{row + row * border_dim}];\n",
                    "    }\n",
                ]
            )
        border_correction_implementations = [
            "".join(
                ["    {\n", f"        double value = x[{core_index}];\n"]
                + [
                    f"        value -= "
                    f"border_solution[{col * dim + core_index}] * theta[{col}];\n"
                    for col in range(border_dim)
                ]
                + [f"        x[{core_index}] = value;\n", "    }\n"]
            )
            for core_index in core_indices
        ]
        border_solve_suffix_lines = []
        border_solve_suffix_lines.extend(
            f"    x[{border_index}] = theta[{col}];\n"
            for col, border_index in enumerate(bordered_x_indices)
        )
        border_solve_prefix = "".join(border_solve_prefix_lines)
        border_solve_middle = "".join(border_solve_middle_lines)
        border_solve_suffix = "".join(border_solve_suffix_lines)

    cpp_header_code = f"""#pragma once

{RESTRICT_MACRO}

namespace {namespace} {{

constexpr int L_nnz = {L_nnz};

constexpr int dim = {dim};
constexpr int core_dim = {core_dim};
constexpr int border_dim = {border_dim};
constexpr int border_solution_size = dim * border_dim;
constexpr int border_factor_size = border_dim * border_dim;
constexpr int x_dim = {x_dim};
constexpr int y_dim = {y_dim};
constexpr int z_dim = {z_dim};
constexpr int expected_positive_inertia = x_dim;
constexpr int expected_negative_inertia = y_dim + z_dim;
constexpr int expected_zero_inertia = 0;

enum class FactorStatus {{
  kSuccess,
  kWrongInertia,
  kZeroPivot,
  kNonFinitePivot,
}};

// Performs an L D L^T decomposition of the matrix (P_MAT * K * P_MAT.T), where
// K = [[ H + r1 I   C.T     G.T    ]
//      [    C     -diag(r2)    0       ]
//      [    G         0    -W - diag(r3)]],
// where:
// 1. H_data is expected to represent np.triu(H) in CSC order.
// 2. C_data and G_data are expected to represent C and G, respectively, in CSC order.
// 3. W is a diagonal matrix, represented by the vector of its diagonal elements, w.
// Returns kSuccess iff the computed factorization has the expected KKT inertia:
// expected_positive_inertia positive pivots and expected_negative_inertia negative pivots.
// NOTE: LT_data, D_inv, border_solution, and border_factor should have sizes
// L_nnz, core_dim, border_solution_size, and border_factor_size, respectively.
FactorStatus ldlt_factor_with_status(const double *SLACG_RESTRICT H_data,
                                      const double *SLACG_RESTRICT C_data,
                                      const double *SLACG_RESTRICT G_data,
                                      const double *SLACG_RESTRICT w,
                                      const double r1,
                                      const double *SLACG_RESTRICT r2,
                                      const double *SLACG_RESTRICT r3,
                                      double *SLACG_RESTRICT LT_data,
                                      double *SLACG_RESTRICT D_inv,
                                      double *SLACG_RESTRICT border_solution,
                                      double *SLACG_RESTRICT border_factor);

// Returns true iff ldlt_factor_with_status returns FactorStatus::kSuccess.
bool ldlt_factor(const double *SLACG_RESTRICT H_data,
                 const double *SLACG_RESTRICT C_data,
                 const double *SLACG_RESTRICT G_data,
                 const double *SLACG_RESTRICT w, const double r1,
                 const double *SLACG_RESTRICT r2,
                 const double *SLACG_RESTRICT r3,
                 double *SLACG_RESTRICT LT_data,
                 double *SLACG_RESTRICT D_inv,
                 double *SLACG_RESTRICT border_solution,
                 double *SLACG_RESTRICT border_factor);

// Solves K * x = b, given a pre-computed L D L^T factorization of (P_MAT * K * P_MAT.T).
// LT_data and D_inv can be computed via the ldlt_factor method defined above.
void ldlt_solve(const double *SLACG_RESTRICT LT_data,
                const double *SLACG_RESTRICT D_inv,
                const double *SLACG_RESTRICT border_solution,
                const double *SLACG_RESTRICT border_factor,
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

    output_name = header_name.rsplit("/", 1)[-1]
    factor_chunk_files = ()
    factor_declarations = ""
    if num_factor_chunks == 1:
        ldlt_implementation = (
            ldlt_prefix
            + "    double D_i;\n"
            + "".join(ldlt_pivot_implementations)
            + ldlt_suffix
        )
    else:
        factor_parameters = (
            ("const double *SLACG_RESTRICT H_data", "H_data"),
            ("const double *SLACG_RESTRICT C_data", "C_data"),
            ("const double *SLACG_RESTRICT G_data", "G_data"),
            ("const double *SLACG_RESTRICT w", "w"),
            ("const double r1", "r1"),
            ("const double *SLACG_RESTRICT r2", "r2"),
            ("const double *SLACG_RESTRICT r3", "r3"),
            ("double *SLACG_RESTRICT LT_data", "LT_data"),
            ("double *SLACG_RESTRICT D_inv", "D_inv"),
            ("double &D_i", "D_i"),
            ("int &positive_count", "positive_count"),
            ("int &negative_count", "negative_count"),
        )
        factor_chunks = _partition_contiguous_by_size(
            ldlt_chunkable_implementations, num_factor_chunks
        )
        declarations = []
        calls = []
        chunk_files = []
        for chunk_index, chunk in enumerate(factor_chunks):
            function_name = f"factor_chunk_{chunk_index}"
            chunk_body = "".join(chunk)
            chunk_parameters = tuple(
                (declaration, name)
                for declaration, name in factor_parameters
                if re.search(rf"\b{re.escape(name)}\b", chunk_body)
            )
            factor_chunk_arguments = ",\n    ".join(
                declaration for declaration, _ in chunk_parameters
            )
            factor_chunk_call_arguments = ", ".join(
                name for _, name in chunk_parameters
            )
            declarations.append(
                f"FactorStatus {function_name}({factor_chunk_arguments});"
            )
            calls.append(
                f"    status = internal::{function_name}("
                f"{factor_chunk_call_arguments});\n"
                "    if (status != FactorStatus::kSuccess) {\n"
                "        return status;\n"
                "    }\n"
            )
            chunk_code = (
                "namespace internal {\n\n"
                f"FactorStatus {function_name}({factor_chunk_arguments}) {{\n"
                + chunk_body
                + "    return FactorStatus::kSuccess;\n"
                "}\n\n"
                "}  // namespace internal"
            )
            chunk_files.append(
                GeneratedFile(
                    f"{output_name}_factor_chunk_{chunk_index}.cpp",
                    _implementation_code(
                        header_name, namespace, ("cmath",), chunk_code
                    ),
                )
            )
        factor_declarations = (
            "namespace internal {\n\n"
            + "\n\n".join(declarations)
            + "\n\n}  // namespace internal\n\n"
        )
        ldlt_implementation = (
            ldlt_prefix
            + "    double D_i;\n"
            + "    FactorStatus status;\n"
            + "".join(calls)
            + ldlt_suffix
        )
        factor_chunk_files = tuple(chunk_files)

    ldlt_implementation += border_factor_implementation
    factor_code = f"""namespace internal {{

void ldlt_solve_core(const double *SLACG_RESTRICT LT_data,
                     const double *SLACG_RESTRICT D_inv,
                     const double *SLACG_RESTRICT b,
                     double *SLACG_RESTRICT x);

}}  // namespace internal

{factor_declarations}FactorStatus ldlt_factor_with_status(
    const double *SLACG_RESTRICT H_data,
    const double *SLACG_RESTRICT C_data,
    const double *SLACG_RESTRICT G_data,
    const double *SLACG_RESTRICT w, const double r1,
    const double *SLACG_RESTRICT r2,
    const double *SLACG_RESTRICT r3,
    double *SLACG_RESTRICT LT_data,
    double *SLACG_RESTRICT D_inv,
    double *SLACG_RESTRICT border_solution,
    double *SLACG_RESTRICT border_factor) {{
{ldlt_implementation}}}

bool ldlt_factor(const double *SLACG_RESTRICT H_data,
                 const double *SLACG_RESTRICT C_data,
                 const double *SLACG_RESTRICT G_data,
                 const double *SLACG_RESTRICT w, const double r1,
                 const double *SLACG_RESTRICT r2,
                 const double *SLACG_RESTRICT r3,
                 double *SLACG_RESTRICT LT_data,
                 double *SLACG_RESTRICT D_inv,
                 double *SLACG_RESTRICT border_solution,
                 double *SLACG_RESTRICT border_factor) {{
    return ldlt_factor_with_status(H_data, C_data, G_data, w, r1, r2, r3,
                                   LT_data, D_inv, border_solution,
                                   border_factor) == FactorStatus::kSuccess;
}}"""

    solve_chunk_files = ()
    if num_solve_chunks == 1:
        solve_helpers = f"""namespace {{
void solve_lower_unitriangular(const double *SLACG_RESTRICT LT_data,
                               const double *SLACG_RESTRICT b,
                               double *SLACG_RESTRICT x) {{
{"".join(solve_lower_unitriangular_implementations)}}}

void solve_upper_unitriangular(const double *SLACG_RESTRICT LT_data,
                               const double *SLACG_RESTRICT b,
                               double *SLACG_RESTRICT x) {{
{"".join(solve_upper_unitriangular_implementations)}}}
}}  // namespace
"""
        solve_lower_calls = (
            "    solve_lower_unitriangular(LT_data, tmp2.data(), tmp1.data());\n"
        )
        solve_upper_calls = (
            "    solve_upper_unitriangular(LT_data, tmp1.data(), tmp2.data());\n"
        )
        permute_b_calls = "".join(permute_b_implementations)
        scale_diagonal_calls = "".join(scale_diagonal_implementations)
        permute_solution_calls = "".join(permute_solution_implementations)
        border_rhs_calls = "".join(border_rhs_implementations)
        border_correction_calls = "".join(border_correction_implementations)
    else:
        permute_b_chunks = _partition_contiguous_by_size(
            permute_b_implementations, num_solve_chunks
        )
        lower_chunks = _partition_contiguous_by_size(
            solve_lower_unitriangular_implementations, num_solve_chunks
        )
        scale_diagonal_chunks = _partition_contiguous_by_size(
            scale_diagonal_implementations, num_solve_chunks
        )
        upper_chunks = _partition_contiguous_by_size(
            solve_upper_unitriangular_implementations, num_solve_chunks
        )
        permute_solution_chunks = _partition_contiguous_by_size(
            permute_solution_implementations, num_solve_chunks
        )
        border_rhs_chunks = _partition_implementations(
            border_rhs_implementations, num_solve_chunks
        )
        border_correction_chunks = _partition_implementations(
            border_correction_implementations, num_solve_chunks
        )
        declarations = []
        permute_b_calls = []
        lower_calls = []
        scale_diagonal_calls = []
        upper_calls = []
        permute_solution_calls = []
        border_rhs_calls = []
        border_correction_calls = []
        chunk_files = []
        solve_chunks = zip(
            permute_b_chunks,
            lower_chunks,
            scale_diagonal_chunks,
            upper_chunks,
            permute_solution_chunks,
            border_rhs_chunks,
            border_correction_chunks,
            strict=True,
        )
        for chunk_index, (
            permute_b_chunk,
            lower_chunk,
            scale_diagonal_chunk,
            upper_chunk,
            permute_solution_chunk,
            border_rhs_chunk,
            border_correction_chunk,
        ) in enumerate(solve_chunks):
            permute_b_name = f"permute_b_chunk_{chunk_index}"
            scale_diagonal_name = f"scale_diagonal_chunk_{chunk_index}"
            permute_solution_name = f"permute_solution_chunk_{chunk_index}"
            border_rhs_name = f"accumulate_border_rhs_chunk_{chunk_index}"
            border_correction_name = f"correct_border_solution_chunk_{chunk_index}"
            triangular_parameters = (
                "const double *SLACG_RESTRICT LT_data,\n"
                "    const double *SLACG_RESTRICT b,\n"
                "    double *SLACG_RESTRICT x"
            )
            declarations.extend(
                [
                    f"void {permute_b_name}(const double *SLACG_RESTRICT b, "
                    "double *SLACG_RESTRICT tmp2);",
                    f"void solve_lower_unitriangular_chunk_{chunk_index}("
                    f"{triangular_parameters});",
                    f"void {scale_diagonal_name}("
                    "const double *SLACG_RESTRICT D_inv, "
                    "double *SLACG_RESTRICT tmp1);",
                    f"void solve_upper_unitriangular_chunk_{chunk_index}("
                    f"{triangular_parameters});",
                    f"void {permute_solution_name}("
                    "const double *SLACG_RESTRICT tmp2, "
                    "double *SLACG_RESTRICT x);",
                ]
            )
            if border_dim > 0:
                declarations.extend(
                    [
                        f"void {border_rhs_name}("
                        "const double *SLACG_RESTRICT border_solution, "
                        "const double *SLACG_RESTRICT b, double *theta);",
                        f"void {border_correction_name}("
                        "const double *SLACG_RESTRICT border_solution, "
                        "const double *theta, double *SLACG_RESTRICT x);",
                    ]
                )
            permute_b_calls.append(f"    internal::{permute_b_name}(b, tmp2.data());\n")
            lower_calls.append(
                f"    internal::solve_lower_unitriangular_chunk_{chunk_index}("
                "LT_data, tmp2.data(), tmp1.data());\n"
            )
            scale_diagonal_calls.append(
                f"    internal::{scale_diagonal_name}(D_inv, tmp1.data());\n"
            )
            upper_calls.append(
                f"    internal::solve_upper_unitriangular_chunk_{chunk_index}("
                "LT_data, tmp1.data(), tmp2.data());\n"
            )
            permute_solution_calls.append(
                f"    internal::{permute_solution_name}(tmp2.data(), x);\n"
            )
            if border_dim > 0:
                border_rhs_calls.append(
                    f"    internal::{border_rhs_name}(border_solution, b, "
                    "theta.data());\n"
                )
                border_correction_calls.append(
                    f"    internal::{border_correction_name}(border_solution, "
                    "theta.data(), x);\n"
                )
            border_chunk_code = ""
            if border_dim > 0:
                border_chunk_code = f"""

void {border_rhs_name}(
    const double *SLACG_RESTRICT border_solution,
    const double *SLACG_RESTRICT b, double *theta) {{
{"".join(border_rhs_chunk)}}}

void {border_correction_name}(
    const double *SLACG_RESTRICT border_solution, const double *theta,
    double *SLACG_RESTRICT x) {{
{"".join(border_correction_chunk)}}}
"""
            chunk_code = f"""namespace internal {{

void {permute_b_name}(const double *SLACG_RESTRICT b,
                      double *SLACG_RESTRICT tmp2) {{
{"".join(permute_b_chunk)}}}

void solve_lower_unitriangular_chunk_{chunk_index}({triangular_parameters}) {{
{"".join(lower_chunk)}}}

void {scale_diagonal_name}(const double *SLACG_RESTRICT D_inv,
                           double *SLACG_RESTRICT tmp1) {{
{"".join(scale_diagonal_chunk)}}}

void solve_upper_unitriangular_chunk_{chunk_index}({triangular_parameters}) {{
{"".join(upper_chunk)}}}

void {permute_solution_name}(const double *SLACG_RESTRICT tmp2,
                             double *SLACG_RESTRICT x) {{
{"".join(permute_solution_chunk)}}}
{border_chunk_code}

}}  // namespace internal"""
            chunk_files.append(
                GeneratedFile(
                    f"{output_name}_solve_chunk_{chunk_index}.cpp",
                    _implementation_code(header_name, namespace, (), chunk_code),
                )
            )
        solve_helpers = (
            "namespace internal {\n\n"
            + "\n\n".join(declarations)
            + "\n\n}  // namespace internal\n"
        )
        permute_b_calls = "".join(permute_b_calls)
        solve_lower_calls = "".join(lower_calls)
        scale_diagonal_calls = "".join(scale_diagonal_calls)
        solve_upper_calls = "".join(upper_calls)
        permute_solution_calls = "".join(permute_solution_calls)
        border_rhs_calls = "".join(border_rhs_calls)
        border_correction_calls = "".join(border_correction_calls)
        solve_chunk_files = tuple(chunk_files)

    solve_code = f"""{solve_helpers}

namespace internal {{

void ldlt_solve_core(const double *SLACG_RESTRICT LT_data,
                     const double *SLACG_RESTRICT D_inv,
                     const double *SLACG_RESTRICT b,
                     double *SLACG_RESTRICT x);

}}  // namespace internal

void internal::ldlt_solve_core(const double *SLACG_RESTRICT LT_data,
                               const double *SLACG_RESTRICT D_inv,
                               const double *SLACG_RESTRICT b,
                               double *SLACG_RESTRICT x) {{
    std::array<double, {core_dim}> tmp1;
    std::array<double, {core_dim}> tmp2;
{permute_b_calls}
{solve_lower_calls}
{scale_diagonal_calls}
{solve_upper_calls}

{permute_solution_calls}}}

void ldlt_solve(const double *SLACG_RESTRICT LT_data,
                const double *SLACG_RESTRICT D_inv,
                const double *SLACG_RESTRICT border_solution,
                const double *SLACG_RESTRICT border_factor,
                const double *SLACG_RESTRICT b,
                double *SLACG_RESTRICT x) {{
    internal::ldlt_solve_core(LT_data, D_inv, b, x);
{border_solve_prefix}
{border_rhs_calls}
{border_solve_middle}
{border_correction_calls}
{border_solve_suffix}}}"""

    product_chunk_files = ()
    if num_product_chunks == 1:
        hessian_declarations = ""
        equality_declarations = ""
        inequality_declarations = ""
        add_H_calls = add_upper_symmetric_Hx_to_y_impl
        add_CTx_calls = add_CTx_to_y_impl
        add_Cx_calls = add_Cx_to_y_impl
        add_fused_C_calls = add_CTx_and_Cx_to_y_impl
        add_GTx_calls = add_GTx_to_y_impl
        add_Gx_calls = add_Gx_to_y_impl
        add_fused_G_calls = add_GTx_and_Gx_to_y_impl
    else:
        product_implementation_groups = tuple(
            zip(
                _partition_implementations(
                    _split_implementation_lines(
                        add_upper_symmetric_Hx_to_y_implementations, 256
                    ),
                    num_product_chunks,
                ),
                _partition_implementations(
                    _split_implementation_lines(add_CTx_to_y_implementations, 256),
                    num_product_chunks,
                ),
                _partition_implementations(
                    _split_implementation_lines(add_Cx_to_y_implementations, 256),
                    num_product_chunks,
                ),
                _partition_implementations(
                    _fused_product_implementations(
                        SPARSE_C,
                        "C",
                        "x_y",
                        "y_y",
                        max_entries_per_block=128,
                    ),
                    num_product_chunks,
                ),
                _partition_implementations(
                    _split_implementation_lines(add_GTx_to_y_implementations, 256),
                    num_product_chunks,
                ),
                _partition_implementations(
                    _split_implementation_lines(add_Gx_to_y_implementations, 256),
                    num_product_chunks,
                ),
                _partition_implementations(
                    _fused_product_implementations(
                        SPARSE_G,
                        "G",
                        "x_z",
                        "y_z",
                        max_entries_per_block=128,
                    ),
                    num_product_chunks,
                ),
                strict=True,
            )
        )
        hessian_declarations = ["namespace internal {\n"]
        equality_declarations = ["namespace internal {\n"]
        inequality_declarations = ["namespace internal {\n"]
        add_H_calls = []
        add_CTx_calls = []
        add_Cx_calls = []
        add_fused_C_calls = []
        add_GTx_calls = []
        add_Gx_calls = []
        add_fused_G_calls = []
        chunk_files = []
        for chunk_index, implementations in enumerate(product_implementation_groups):
            (
                add_H_impl,
                add_CTx_impl,
                add_Cx_impl,
                add_fused_C_impl,
                add_GTx_impl,
                add_Gx_impl,
                add_fused_G_impl,
            ) = implementations
            hessian_declarations.extend(
                [
                    f"void add_H_chunk_{chunk_index}(const double *, "
                    "const double *, double *);\n",
                    f"void add_fused_C_chunk_{chunk_index}(const double *, "
                    "const double *, const double *, double *, double *);\n",
                    f"void add_fused_G_chunk_{chunk_index}(const double *, "
                    "const double *, const double *, double *, double *);\n",
                ]
            )
            equality_declarations.extend(
                [
                    f"void add_CTx_chunk_{chunk_index}(const double *, "
                    "const double *, double *);\n",
                    f"void add_Cx_chunk_{chunk_index}(const double *, "
                    "const double *, double *);\n",
                ]
            )
            inequality_declarations.extend(
                [
                    f"void add_GTx_chunk_{chunk_index}(const double *, "
                    "const double *, double *);\n",
                    f"void add_Gx_chunk_{chunk_index}(const double *, "
                    "const double *, double *);\n",
                ]
            )
            add_H_calls.append(
                f"    internal::add_H_chunk_{chunk_index}(H_data, x, y);\n"
            )
            add_CTx_calls.append(
                f"    internal::add_CTx_chunk_{chunk_index}(C_data, x, y);\n"
            )
            add_Cx_calls.append(
                f"    internal::add_Cx_chunk_{chunk_index}(C_data, x, y);\n"
            )
            add_fused_C_calls.append(
                f"    internal::add_fused_C_chunk_{chunk_index}("
                "C_data, x_x, x_y, y_x, y_y);\n"
            )
            add_GTx_calls.append(
                f"    internal::add_GTx_chunk_{chunk_index}(G_data, x, y);\n"
            )
            add_Gx_calls.append(
                f"    internal::add_Gx_chunk_{chunk_index}(G_data, x, y);\n"
            )
            add_fused_G_calls.append(
                f"    internal::add_fused_G_chunk_{chunk_index}("
                "G_data, x_x, x_z, y_x, y_z);\n"
            )
            chunk_code = f"""namespace internal {{

void add_H_chunk_{chunk_index}(const double *SLACG_RESTRICT H_data,
                               const double *SLACG_RESTRICT x,
                               double *SLACG_RESTRICT y) {{
{add_upper_symmetric_Hx_to_y_prefix}{"".join(add_H_impl)}}}

void add_CTx_chunk_{chunk_index}(const double *SLACG_RESTRICT C_data,
                                 const double *SLACG_RESTRICT x,
                                 double *SLACG_RESTRICT y) {{
{add_CTx_to_y_prefix}{"".join(add_CTx_impl)}}}

void add_Cx_chunk_{chunk_index}(const double *SLACG_RESTRICT C_data,
                                const double *SLACG_RESTRICT x,
                                double *SLACG_RESTRICT y) {{
{add_Cx_to_y_prefix}{"".join(add_Cx_impl)}}}

void add_fused_C_chunk_{chunk_index}(const double *SLACG_RESTRICT C_data,
                                     const double *SLACG_RESTRICT x_x,
                                     const double *SLACG_RESTRICT x_y,
                                     double *SLACG_RESTRICT y_x,
                                     double *SLACG_RESTRICT y_y) {{
{add_CTx_and_Cx_to_y_prefix}{"".join(add_fused_C_impl)}}}

void add_GTx_chunk_{chunk_index}(const double *SLACG_RESTRICT G_data,
                                 const double *SLACG_RESTRICT x,
                                 double *SLACG_RESTRICT y) {{
{add_GTx_to_y_prefix}{"".join(add_GTx_impl)}}}

void add_Gx_chunk_{chunk_index}(const double *SLACG_RESTRICT G_data,
                                const double *SLACG_RESTRICT x,
                                double *SLACG_RESTRICT y) {{
{add_Gx_to_y_prefix}{"".join(add_Gx_impl)}}}

void add_fused_G_chunk_{chunk_index}(const double *SLACG_RESTRICT G_data,
                                     const double *SLACG_RESTRICT x_x,
                                     const double *SLACG_RESTRICT x_z,
                                     double *SLACG_RESTRICT y_x,
                                     double *SLACG_RESTRICT y_z) {{
{add_GTx_and_Gx_to_y_prefix}{"".join(add_fused_G_impl)}}}

}}  // namespace internal"""
            chunk_files.append(
                GeneratedFile(
                    f"{output_name}_product_chunk_{chunk_index}.cpp",
                    _implementation_code(header_name, namespace, (), chunk_code),
                )
            )
        hessian_declarations.append("}  // namespace internal\n\n")
        equality_declarations.append("}  // namespace internal\n\n")
        inequality_declarations.append("}  // namespace internal\n\n")
        hessian_declarations = "".join(hessian_declarations)
        equality_declarations = "".join(equality_declarations)
        inequality_declarations = "".join(inequality_declarations)
        add_H_calls = "".join(add_H_calls)
        add_CTx_calls = "".join(add_CTx_calls)
        add_Cx_calls = "".join(add_Cx_calls)
        add_fused_C_calls = "".join(add_fused_C_calls)
        add_GTx_calls = "".join(add_GTx_calls)
        add_Gx_calls = "".join(add_Gx_calls)
        add_fused_G_calls = "".join(add_fused_G_calls)
        product_chunk_files = tuple(chunk_files)

    hessian_and_kkt_product_code = f"""{hessian_declarations}void add_upper_symmetric_Hx_to_y(
    const double *SLACG_RESTRICT H_data,
    const double *SLACG_RESTRICT x,
    double *SLACG_RESTRICT y) {{
{add_H_calls}}}

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

{add_fused_C_calls}
{add_fused_G_calls}
    for (int i = 0; i < {x_dim}; ++i) {{
        y_x[i] += r1 * x_x[i];
    }}

    for (int i = 0; i < {y_dim}; ++i) {{
      y_y[i] -= r2[i] * x_y[i];
    }}

    for (int i = 0; i < {z_dim}; ++i) {{
      y_z[i] -= (w[i] + r3[i]) * x_z[i];
    }}
}}"""

    equality_product_code = f"""{equality_declarations}void add_CTx_to_y(
    const double *SLACG_RESTRICT C_data,
    const double *SLACG_RESTRICT x,
    double *SLACG_RESTRICT y) {{
{add_CTx_calls}}}

void add_Cx_to_y(const double *SLACG_RESTRICT C_data,
                 const double *SLACG_RESTRICT x,
                 double *SLACG_RESTRICT y) {{
{add_Cx_calls}}}"""

    inequality_product_code = f"""{inequality_declarations}void add_GTx_to_y(
    const double *SLACG_RESTRICT G_data,
    const double *SLACG_RESTRICT x,
    double *SLACG_RESTRICT y) {{
{add_GTx_calls}}}

void add_Gx_to_y(const double *SLACG_RESTRICT G_data,
                 const double *SLACG_RESTRICT x,
                 double *SLACG_RESTRICT y) {{
{add_Gx_calls}}}"""

    return (
        GeneratedFile(f"{output_name}.hpp", cpp_header_code),
        GeneratedFile(
            f"{output_name}_factor.cpp",
            _implementation_code(
                header_name,
                namespace,
                tuple(
                    header
                    for header, needed in (
                        ("array", border_dim > 0),
                        ("cmath", num_factor_chunks == 1 or border_dim > 0),
                    )
                    if needed
                ),
                factor_code,
            ),
        ),
        GeneratedFile(
            f"{output_name}_solve.cpp",
            _implementation_code(header_name, namespace, ("array",), solve_code),
        ),
        GeneratedFile(
            f"{output_name}_hessian_and_kkt_product.cpp",
            _implementation_code(
                header_name, namespace, (), hessian_and_kkt_product_code
            ),
        ),
        GeneratedFile(
            f"{output_name}_equality_product.cpp",
            _implementation_code(header_name, namespace, (), equality_product_code),
        ),
        GeneratedFile(
            f"{output_name}_inequality_product.cpp",
            _implementation_code(header_name, namespace, (), inequality_product_code),
        ),
        *factor_chunk_files,
        *solve_chunk_files,
        *product_chunk_files,
    )
