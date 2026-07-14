_KKT_SOURCE_SUFFIXES = [
    "_factor.cpp",
    "_solve.cpp",
    "_hessian_and_kkt_product.cpp",
    "_equality_product.cpp",
    "_inequality_product.cpp",
]

def kkt_codegen_sources(name, num_factor_chunks = 1):
    if num_factor_chunks < 1:
        fail("num_factor_chunks must be positive")
    sources = [name + suffix for suffix in _KKT_SOURCE_SUFFIXES]
    if num_factor_chunks > 1:
        sources.extend([
            name + "_factor_chunk_{}.cpp".format(i)
            for i in range(num_factor_chunks)
        ])
    return sources

def kkt_codegen_files(name, num_factor_chunks = 1):
    return [name + ".hpp"] + kkt_codegen_sources(name, num_factor_chunks)
