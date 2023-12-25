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
This submodule defines an optimization function that applies a sequential rotation of
one of the arrays against the other around an axis
So as to minimize the "Steric interaction" loss function
"""

from .. import MOLLI_USING_EXTENSIONS
import numpy as np
from numpy.typing import ArrayLike
from .rotation import rotation_matrix_from_axis


def _optimize_rotation(
    c1: ArrayLike, c2: ArrayLike, ax: ArrayLike, resolution: int = 12
):
    if not MOLLI_USING_EXTENSIONS:
        raise NotImplementedError
    else:
        import molli_xt as xt

    angles = np.radians(np.arange(0, 360, step=360 // resolution))
    aug_c2 = np.array([c2 @ rotation_matrix_from_axis(ax, ang) for ang in angles])

    dist = (
        xt.cdist32_eu2(aug_c2, c1) + 0.05
    )  # This latter part is damping to avoid singularities when and if present
    loss = np.sum(1 / dist, axis=(1, 2))

    best_angle = angles[np.argmin(loss)]
    return rotation_matrix_from_axis(ax, best_angle)
