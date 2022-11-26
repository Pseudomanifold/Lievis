import numpy as np
import matplotlib.pyplot as plt

from lievis.metrics import frobenius_distance

from geomstats.geometry.general_linear import GeneralLinear
from sklearn.manifold import MDS

import numba

@numba.njit
def pairwise_numba(X):
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = frobenius_distance(X[i], X[j])
            D[j, i] = D[i, j]

    return D

if __name__ == "__main__":
    n = 500
    d = 4

    rng = np.random.default_rng(42)

    X = []
    y = []

    group = GeneralLinear(d)

    X = group.random_point(n)
    y = np.asarray([np.linalg.det(M) for M in X])
    #X = X.reshape(n, -1)

    emb = MDS(dissimilarity="precomputed")
    #D = group.metric.dist_pairwise(X)
    D = pairwise_numba(X)

    X_emb = emb.fit_transform(D)
    print(y.shape)

    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=y)
    plt.colorbar()
    plt.show()
