import numpy as np
import scipy as sp


def mat_vec_mult_codegen(M, namespace, header_name):
    assert len(M.shape) == 2

    cpp_header_code = f"""
    #pragma once

    namespace {namespace}
    {{

    """

    cpp_impl_code = f"""
    #include "{header_name}.hpp"

    #include <array>

    namespace {namespace}
    {{

    """

    M_is_symmetric = M.shape[0] == M.shape[1] and (M == M.T).all()

    if M_is_symmetric:
        SPARSE_UPPER_M = sp.sparse.csc_matrix(np.triu((M != 0.0)))

        add_upper_symmetric_Ax_to_y_impl = ""

        for j in range(M.shape[1]):
            for k in range(
                SPARSE_UPPER_M.indptr[j], SPARSE_UPPER_M.indptr[j + 1]
            ):
                i = SPARSE_UPPER_M.indices[k]
                add_upper_symmetric_Ax_to_y_impl += (
                    f"y[{i}] += A_data[{k}] * x[{j}];\n"
                )
                if i != j:
                    add_upper_symmetric_Ax_to_y_impl += (
                        f"y[{j}] += A_data[{k}] * x[{i}];\n"
                    )

        cpp_header_code += """
            // Performs y += A @ x, where A_data is expected to represent np.triu(A) in CSC order.
            void add_upper_symmetric_Ax_to_y(const double* A_data, const double* x, double* y);
        """

        cpp_impl_code += f"""
            void add_upper_symmetric_Ax_to_y(const double* A_data, const double* x, double* y) {{
            {add_upper_symmetric_Ax_to_y_impl}
            }}
        """

    else:
        SPARSE_M = sp.sparse.csc_matrix((M != 0.0))

        add_Ax_to_y_impl = ""

        add_ATx_to_y_impl = ""

        for j in range(M.shape[1]):
            for k in range(SPARSE_M.indptr[j], SPARSE_M.indptr[j + 1]):
                i = SPARSE_M.indices[k]
                add_Ax_to_y_impl += f"y[{i}] += A_data[{k}] * x[{j}];\n"
                add_ATx_to_y_impl += f"y[{j}] += A_data[{k}] * x[{i}];\n"

        cpp_header_code += """
            // Performs y += A @ x, where A_data is expected to be in CSC order.
            void add_Ax_to_y(const double* A_data, const double* x, double* y);

            // Performs y += A.T @ x, where A_data is expected to be in CSC order.
            void add_ATx_to_y(const double* A_data, const double* x, double* y);
        """

        cpp_impl_code += f"""
            void add_Ax_to_y(const double* A_data, const double* x, double* y) {{
            {add_Ax_to_y_impl}
            }}

            void add_ATx_to_y(const double* A_data, const double* x, double* y) {{
            {add_ATx_to_y_impl}
            }}
        """

    cpp_header_code += f"""

    }} // namespace {namespace}
    """

    cpp_impl_code += f"""
    }} // namespace {namespace}
    """

    return cpp_header_code, cpp_impl_code
