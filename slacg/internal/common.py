import numpy as np
import scipy as sp


def build_sparse_LT(M, P):
    n = M.shape[0]

    P_MAT = np.zeros_like(M)
    P_MAT[np.arange(n), P] = 1.0

    N = P_MAT @ M @ P_MAT.T

    L = np.tril(N, k=-1) != 0.0

    # Note that The L D L^T decomposition of N can be computed with the following recursion:
    # D_i    = N_ii - sum_{j=0}^{i-1} L_{ij}^2 D_j
    # L_{ij} = (1 / D_j) * (N_{ij} - sum_{k=0}^{j-1} L_{ik} L_{jk} D_k)

    for i in range(n):
        for j in range(i):
            L[i, j] = L[i, j] or np.any(np.logical_and(L[i, :j], L[j, :j]))

    return sp.sparse.csc_matrix(L.T)
