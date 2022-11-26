"""Different metrics for points on manifolds."""

import numba
import numpy as np


@numba.njit()
def frobenius_distance(A, B):
    return np.linalg.norm(A-B)
