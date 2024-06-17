# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# The only function is implemented according to the following publication
# joriki (https://math.stackexchange.com/users/6622/joriki),
# Best Fitting Plane given a Set of Points, URL (version: 2021-05-10):
# https://math.stackexchange.com/q/99317
# ================================================================================


"""
This submodule deals with calculation of a mean plane
"""

import numpy as np
from numpy.typing import ArrayLike


def mean_plane(_vecs: ArrayLike) -> np.ndarray:
    """
    Finds best fitting plane for a given set of N points in a 3D space and returns
    normal vector to it.

    Parameters:
    -----------
    _vec: ArrayLike
        Set of points in a 3D space of size (n_points, 3)

    Returns:
    --------
    numpy.ndarray
        Normal vector to best fitting plane

    Examples:
    ---------
    >>> import molli as ml
    >>> points = [ [2.3, -1.8, 4.5], [-3.1, 0.7, 2.9], [1.5, 3.2, -0.6], [0.8, -2.5, 1.4], [-4.0, 1.9, -3.7]]
    >>> ml.math.mean_plane(points)
    array([-0.12846037,  0.79304011,  0.59547067])

    Notes:
    ---------
    Inspired by:
    https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points

    """

    centered = np.array(_vecs)
    assert (
        centered.ndim == 2 and centered.shape[-1] == 3
    ), "set of points should be of shape (n_points, 3)"
    centered -= np.average(centered, axis=0)
    return np.linalg.svd(centered.T)[0][:, -1]
