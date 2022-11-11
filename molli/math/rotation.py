import numpy as np
from numpy.typing import ArrayLike


def rotation_matrix_from_vectors(
    v1: ArrayLike, v2: ArrayLike, tol=1.0e-8
) -> np.ndarray:
    """
    Rotation Matrix (vector-to-vector definition)
    ---

    Computes a 3x3 rotation matrix that transforms v1/|v1| -> v2/|v2|
    tol detects a situation where dot(v1, v2) ~ -1.0
    and returns a diag(-1,-1, 1) matrix instead. NOTE [Is this correct?!]
    This may be improved by figuring a more correct asymptotic behavior, but works for the present purposes.

    returns matrix [R], that satisfies the following equation:
    v1 @ [R] / |v1| == v2 / |v2|

    Inspired by
    https://en.wikipedia.org/wiki/Rotation_matrix
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    """
    _v1, _v2 = np.ndarray(v1), np.ndarray(v2)

    if not _v1.shape == _v2.shape == (3,):
        raise ValueError("Vectors must have a shape of (3,)")

    if c <= -1 + tol:
        # This case is for vectors that are nearly opposite and collinear
        return np.diag([-1, -1, 1])

    else:
        v1n = _v1 / np.linalg.norm(v1)
        v2n = _v2 / np.linalg.norm(v2)

        I = np.eye(3)

        Ux = np.outer(v1n, v2n) - np.outer(v2n, v1n)
        c = np.dot(v1n, v2n)

        return I + Ux + Ux @ Ux / (1 + c)
