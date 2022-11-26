import numpy as np
import matplotlib.pyplot as plt

from lievis.metrics import frobenius_distance

from geomstats.geometry.general_linear import GeneralLinear
from sklearn.manifold import MDS

import numba
import umap

@numba.njit
def pairwise_numba(X):
    n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = frobenius_distance(X[i], X[j])
            D[j, i] = D[i, j]

    return D

if __name__ == "__main__":
    n = 100
    d = 10

    rng = np.random.default_rng(42)

    X = []
    z = []

    for dim in range(2, d + 1):
        group = GeneralLinear(dim)
        sample = group.random_point(n)

        if dim != d:
            sample = np.asarray([
                np.block([
                    [M, np.zeros((dim, d - dim))],
                    [np.zeros((d - dim, dim)), np.eye(d - dim)]
                ]) for M in sample
            ])

        X.append(sample)
        z.append([dim] * n)

    X = np.asarray(X)
    X = X.reshape(-1, d, d)
    y = np.asarray([np.linalg.norm(M) for M in X])
    z = np.asarray(z).ravel()

    emb = umap.UMAP(metric="precomputed")
    #D = group.metric.dist_pairwise(X)
    D = pairwise_numba(X)

    X_emb = emb.fit_transform(D)
    print(y.shape)

    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=z)
    plt.colorbar()
    plt.show()
