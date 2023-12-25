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


def mean_plane(_vecs: ArrayLike):
    """
    https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    """
    centered = np.array(_vecs)
    centered -= np.average(centered, axis=0)
    return np.linalg.svd(centered.T)[0][:, -1]
