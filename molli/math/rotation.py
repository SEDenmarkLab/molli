# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Alexander S. Shved <shvedalx@illinois.edu>
#
# S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign
# https://denmarkgroup.illinois.edu/
#
# Copyright 2022-2023 The Board of Trustees of the University of Illinois.
# All Rights Reserved.
#
# Licensed under the terms MIT License
# The License is included in the distribution as LICENSE file.
# You may not use this file except in compliance with the License.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# ================================================================================


"""
This file contains useful functions that define rotation matrices.
"""

import numpy as np
from numpy.typing import ArrayLike
import math


def rotation_matrix_from_vectors(
    _v1: ArrayLike, _v2: ArrayLike, tol=1.0e-8
) -> np.ndarray:
    """
    Rotation Matrix (vector-to-vector definition)
    ---

    Computes a 3x3 rotation matrix that transforms `v1/|v1| -> v2/|v2|`
    tol detects a situation where dot(v1, v2) ~ -1.0
    and returns a diag(-1,-1, 1) matrix instead. NOTE: [Is this correct?!]
    This may be improved by figuring a more correct asymptotic behavior, but works for the present purposes.

    returns matrix [R], that satisfies the following equation:
    `v1 @ [R] / |v1| == v2 / |v2|`

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
        # Here we determine a vector that is orthogonal to v2n
        # So instead of one -180 degree rotation we do two rotations
        # One to an orthogonal vector `ort`
        _rcp = 1.0
        while abs(_rcp) > 0.01:  # In most tested cases it converges within one step
            RV = np.random.rand(3)
            RV /= np.linalg.norm(RV)
            ort = RV - v2n * np.dot(RV, v2n)  # This is a Gram-Schmidt orthogonalization
            _rcp = np.dot(ort, v2n)
        return rotation_matrix_from_vectors(v1, ort) @ rotation_matrix_from_vectors(
            ort, v2
        )
    else:
        I = np.eye(3)
        Ux = np.outer(v1n, v2n) - np.outer(v2n, v1n)
        return I + Ux + Ux @ Ux / (1 + c)


def rotation_matrix_from_axis(_axis: ArrayLike, angle: float) -> np.ndarray:
    """
    Calculates matrix for rotation around `_axis` vector using Rodrigues rotation formula:
        `rotation_matrix = I + sin(phi) * W + (1-cos(phi)) W @ W`,
    where `I` is the identity matrix, `phi` is the rotation angle and
        `W = [ [0, -az, ay], [az, 0, -ax], [-ay, ax, 0] ]`
        represents the skew symmetriac matrix corresponding to the normalized `_axis` vector.

    Parameters:
    -----------
    `_axis`: ArrayLike
        Array-like object representing an axis of rotation

    angle: float
        Rotation angle in radians

    Examples:
    ---------
    >>> import numpy as np
    >>> import molli as ml
    >>> axis_x = [1, 0, 0]
    >>> angle_1 = np.radians(45)
    >>> print(rotation_matrix_from_axis(axis_x, angle_1))
    [[ 1.          0.          0.        ]
     [ 0.          0.70710678 -0.70710678]
     [ 0.          0.70710678  0.70710678]]

    Returns:
    --------
    np.ndarray
        Rotation matrix of shape (3,3)

    Notes:
    -----------
    Inspired by:
    https://mathworld.wolfram.com/RodriguesRotationFormula.html
    https://math.stackexchange.com/questions/142821/matrix-for-rotation-around-a-vector
    """
    # normalizing _axis by dividing it by its Euclidean norm (magnitude):
    ax, ay, az = np.array(_axis) / np.linalg.norm(_axis)

    # constructing a 3x3 matrix W:
    W = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])

    # getting sine and cosine of the rotation angle:
    k1 = math.sin(angle)
    k2 = 1 - math.cos(angle)

    # gathering all into Rodrigues rotation formula:
    return np.eye(3) + k1 * W + k2 * (W @ W)


def rotate_2dvec_outa_plane(
    _vec: ArrayLike, angle: float, _plane_normal: ArrayLike = [0, 0, 1]
):
    """
    Performs a 2D rotation of a vector `_vec` out of a specified plane defined
    by a normal vector `_plane_normal`. The rotation is done by first aligning
    the plane to the xy-plane, performing the rotation in the xy-plane, and
    then transforming the result back to the original plane

    Parameters:
    -----------
    `_vec`: ArrayLike
        Array-like object representing an vector to rotate

    angle: float
        Rotation angle in radians

    `_plane_normal`: ArrayLike
        Normal vector of the plane about which the rotation is performed,
        defaulting to the z-axis if not provided

    Returns:
    --------
    np.ndarray
        Rotation matrix of shape (3,3)

    """
    R = rotation_matrix_from_vectors(_plane_normal, [0, 0, 1])
    ax = np.cross([0, 0, 1], _vec)
    Rinv = np.linalg.inv(R)
    return R @ rotation_matrix_from_axis(ax, angle) @ Rinv
