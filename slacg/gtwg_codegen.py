import scipy as sp

from slacg.internal.common import RESTRICT_MACRO


def gtwg_codegen(G, namespace, header_name):
    assert sp.sparse.issparse(G)
    assert len(G.shape) == 2

    SPARSE_G = G.tocsc(copy=True)
    SPARSE_G.eliminate_zeros()

    GTG = SPARSE_G.T @ SPARSE_G

    SPARSE_GTG = sp.sparse.triu(GTG, format="csc")

    # Maps (i, j), where i <= j, to the corresponding output coordinate.
    GTG_COORDINATE_MAP = {}
    for j in range(SPARSE_GTG.shape[1]):
        for k in range(SPARSE_GTG.indptr[j], SPARSE_GTG.indptr[j + 1]):
            i = int(SPARSE_GTG.indices[k])
            GTG_COORDINATE_MAP[(i, j)] = k

    # Maps each row h to the nonzero entries (j, G_data index) in that row.
    G_NZ_PER_ROW = [[] for _ in range(G.shape[0])]
    for j in range(G.shape[1]):
        for k in range(SPARSE_G.indptr[j], SPARSE_G.indptr[j + 1]):
            i = int(SPARSE_G.indices[k])
            G_NZ_PER_ROW[i].append((j, k))

    for row in G_NZ_PER_ROW:
        row.sort()

    cpp_header_code = f"""
#pragma once

{RESTRICT_MACRO}

namespace {namespace} {{
"""

    cpp_impl_code = f"""
#include "{header_name}.hpp"

#include <array>

namespace {namespace} {{
"""

    gt_w_g_impl = ""

    if SPARSE_GTG.nnz == 0:
        gt_w_g_impl += (
            "    (void) G_data;\n"
            "    (void) w;\n"
            "    (void) r;\n"
            "    (void) gt_w_g;\n"
        )
    else:
        for k in range(SPARSE_GTG.nnz):
            gt_w_g_impl += f"    double gt_w_g_{k} = 0.0;\n"

        for h, row in enumerate(G_NZ_PER_ROW):
            if not row:
                continue
            gt_w_g_impl += f"    const double scale_{h} = w[{h}] + r;\n"
            for a, (i, G_hi) in enumerate(row):
                for j, G_hj in row[a:]:
                    assert i <= j
                    assert (i, j) in GTG_COORDINATE_MAP
                    output_idx = GTG_COORDINATE_MAP[(i, j)]
                    gt_w_g_impl += (
                        f"    gt_w_g_{output_idx} += "
                        f"G_data[{G_hi}] * scale_{h} * G_data[{G_hj}];\n"
                    )

        for k in range(SPARSE_GTG.nnz):
            gt_w_g_impl += f"    gt_w_g[{k}] = gt_w_g_{k};\n"

    cpp_header_code += """
// Computes G.T @ (W + r I) @ G in CSC format, where:
// 1. G_data is expected to represent G in CSC order.
// 2. W is a diagonal matrix, represented by the vector of its diagonal elements, w.
void gt_w_g(const double* SLACG_RESTRICT G_data,
            const double* SLACG_RESTRICT w,
            const double r,
            double* SLACG_RESTRICT gt_w_g);
"""

    cpp_impl_code += f"""
void gt_w_g(const double* SLACG_RESTRICT G_data,
            const double* SLACG_RESTRICT w,
            const double r,
            double* SLACG_RESTRICT gt_w_g) {{
{gt_w_g_impl}}}
"""

    cpp_header_code += f"""
}} // namespace {namespace}
"""

    cpp_impl_code += f"""
}} // namespace {namespace}
"""

    return cpp_header_code, cpp_impl_code
