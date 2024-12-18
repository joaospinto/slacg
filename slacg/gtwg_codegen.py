import numpy as np
import scipy as sp


def gtwg_codegen(G, namespace, header_name):
    assert len(G.shape) == 2

    SPARSE_G = sp.sparse.csc_matrix(G)

    GTG = G.T @ G

    SPARSE_GTG = sp.sparse.csc_matrix(np.triu(GTG))

    # Maps (i, j) to the index of G.data representing G[i, j].
    G_COORDINATE_MAP = {}
    for j in range(G.shape[1]):
        for k in range(SPARSE_G.indptr[j], SPARSE_G.indptr[j + 1]):
            i = SPARSE_G.indices[k]
            G_COORDINATE_MAP[(i, j)] = k

    # Maps (j, k), where j >= k, to {i | G_ij != 0 and G_ik != 0}
    GTG_MAP = {}
    for j in range(G.shape[1]):
        for k in range(j + 1):
            i_nz = [i for i in range(G.shape[0]) if G[i, j] != 0.0 and G[i, k] != 0.0]
            GTG_MAP[(j, k)] = i_nz
            GTG_MAP[(k, j)] = i_nz

    cpp_header_code = f"""
#pragma once

namespace {namespace} {{
"""

    cpp_impl_code = f"""
#include "{header_name}.hpp"

#include <array>

namespace {namespace} {{
"""

    gt_w_g_impl = ""

    for j in range(SPARSE_GTG.shape[1]):
        for k in range(SPARSE_GTG.indptr[j], SPARSE_GTG.indptr[j + 1]):
            line = f"    gt_w_g[{k}] = 0.0"
            i = SPARSE_GTG.indices[k]
            for h in GTG_MAP[(i, j)]:
                G_hi = G_COORDINATE_MAP[(h, i)]
                G_hj = G_COORDINATE_MAP[(h, j)]
                line += f" + G_data[{G_hi}] * (w[{h}] + r) * G_data[{G_hj}]"
            line += ";\n"
            gt_w_g_impl += line

    cpp_header_code += """
// Computes G.T @ (W + r I) @ G in CSC format, where:
// 1. G_data is expected to represent G in CSC order.
// 2. W is a diagonal matrix, represented by the vector of its diagonal elements, w.
void gt_w_g(const double* G_data, const double* w, const double r, double* gt_w_g);
"""

    cpp_impl_code += f"""
void gt_w_g(const double* G_data, const double* w, const double r, double* gt_w_g) {{
{gt_w_g_impl}}}
"""

    cpp_header_code += f"""
}} // namespace {namespace}
"""

    cpp_impl_code += f"""
}} // namespace {namespace}
"""

    return cpp_header_code, cpp_impl_code
