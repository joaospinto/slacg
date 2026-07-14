_KKT_SOURCE_SUFFIXES = [
    "_factor.cpp",
    "_solve.cpp",
    "_hessian_and_kkt_product.cpp",
    "_equality_product.cpp",
    "_inequality_product.cpp",
]

def _validate_num_chunks(value, parameter_name):
    if value < 1:
        fail("{} must be positive".format(parameter_name))

def kkt_codegen_sources(
        name,
        num_factor_chunks = 1,
        num_solve_chunks = 1,
        num_product_chunks = 1):
    _validate_num_chunks(num_factor_chunks, "num_factor_chunks")
    _validate_num_chunks(num_solve_chunks, "num_solve_chunks")
    _validate_num_chunks(num_product_chunks, "num_product_chunks")
    sources = [name + suffix for suffix in _KKT_SOURCE_SUFFIXES]
    if num_factor_chunks > 1:
        sources.extend([
            name + "_factor_chunk_{}.cpp".format(i)
            for i in range(num_factor_chunks)
        ])
    if num_solve_chunks > 1:
        sources.extend([
            name + "_solve_chunk_{}.cpp".format(i)
            for i in range(num_solve_chunks)
        ])
    if num_product_chunks > 1:
        sources.extend([
            name + "_product_chunk_{}.cpp".format(i)
            for i in range(num_product_chunks)
        ])
    return sources

def kkt_codegen_files(
        name,
        num_factor_chunks = 1,
        num_solve_chunks = 1,
        num_product_chunks = 1):
    return [name + ".hpp"] + kkt_codegen_sources(
        name,
        num_factor_chunks,
        num_solve_chunks,
        num_product_chunks,
    )
