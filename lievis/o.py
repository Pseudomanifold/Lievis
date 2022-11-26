import numpy as np
import matplotlib.pyplot as plt

from umap import UMAP

from lievis.metrics import frobenius_distance


if __name__ == "__main__":
    n = 5000
    d = 5

    rng = np.random.default_rng(42)

    X = []
    y = []

    for i in range(n):
        M = rng.normal(size=(d, d))
        Q, *_ = np.linalg.qr(M)

        X.append(Q.ravel())
        y.append(np.linalg.norm(Q))

    X = np.asarray(X)
    y = np.asarray(y)

    emb = UMAP(metric=frobenius_distance)
    X_emb = emb.fit_transform(X)

    plt.scatter(X_emb[:, 0], X[:, 1], c=y)
    plt.colorbar()
    plt.show()
