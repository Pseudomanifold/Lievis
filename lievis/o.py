import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import MDS

from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from lievis.metrics import frobenius_distance

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
    d = 5

    rng = np.random.default_rng(42)

    group = SpecialOrthogonal(n=d)
    X = group.random_point(n)
    y = np.asarray([np.linalg.norm(M) for M in X])

    emb = MDS(dissimilarity="precomputed")
    D = pairwise_numba(X)
    X_emb = emb.fit_transform(D)

    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=y)
    plt.colorbar()
    plt.show()
