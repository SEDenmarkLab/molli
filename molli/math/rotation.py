import numpy as np
from numpy.typing import ArrayLike
import math


def rotation_matrix_from_vectors(
    _v1: ArrayLike, _v2: ArrayLike, tol=1.0e-8
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
    v1 = np.array(_v1)
    v2 = np.array(_v2)

    if not v1.shape == v2.shape == (3,):
        raise ValueError("Vectors must have a shape of (3,)")

    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    c = np.dot(v1n, v2n)

    if c <= -1 + tol:
        # This case is for vectors that are nearly opposite and collinear
        return np.diag([-1, -1, 1])
    else:
        I = np.eye(3)
        Ux = np.outer(v1n, v2n) - np.outer(v2n, v1n)
        return I + Ux + Ux @ Ux / (1 + c)


def rotation_matrix_from_axis(_axis: ArrayLike, angle: float):
    """
    refs
    https://mathworld.wolfram.com/RodriguesRotationFormula.html
    https://math.stackexchange.com/questions/142821/matrix-for-rotation-around-a-vector
    """
    ax, ay, az = np.array(_axis) / np.linalg.norm(_axis)

    W = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])

    k1 = math.sin(angle)
    k2 = 1 - math.cos(angle)

    return np.eye(3) + k1 * W + k2 * (W @ W)
