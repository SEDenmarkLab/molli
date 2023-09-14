"""
This submodule deals with calculation of a mean plane
"""
import numpy as np
from numpy.typing import ArrayLike


def mean_plane(_vecs: ArrayLike):
    """
    https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    """
    centered = np.array(_vecs)
    centered -= np.average(centered, axis=0)
    return np.linalg.svd(centered.T)[0][:, -1]
