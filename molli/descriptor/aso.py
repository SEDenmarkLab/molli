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
# `molli.descriptor.aso`
This module defines the functions necessary to calculate ASO
(Average Steric Occupancy Descriptor)

See 
Zahrt AF, Henle JJ, Rose BT, Wang Y, Darrow WT, Denmark SE. Science 2019, 363 (6424), eaau5631.
Henle JJ, Zahrt AF, Rose BT, Darrow WT, Wang Y, Denmark SE. Journal of the American Chemical Society 2020, 142 (26), 11578-11592.
for definitions
"""

from ..chem import Molecule, ConformerEnsemble
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
from math import dist
import molli_xt
import deprecated


def _where_distance(p, g: np.ndarray, r: float):
    """
    Return a numpy array of where point is within radius from gridpoints
    """

    diff = g - p
    dists = np.sum(diff**2, axis=-1)

    return dists <= r**2


@deprecated.deprecated(
    "This function is left for compatibility and testing purposes only."
    "While it is more readable, due to pure Python for loops it is relatively slow."
)
def aso1(
    ens: ConformerEnsemble,
    grid: np.ndarray,
    weighted: bool = False,
) -> np.ndarray:
    aso_full = np.empty((ens.n_conformers, grid.shape[0]))

    # Iterate over conformers in the ensemble
    # Complexity (O(n_confs * n_gpts))
    for i, c in enumerate(ens):
        # Iterate over atoms
        for j, a in enumerate(c.atoms):
            aso_full[i][_where_distance(c.coords[j], grid, a.vdw_radius)] = 1

    return np.average(
        aso_full,
        axis=0,
        weights=ens.weights if weighted else None,
    )


def aso2(
    ens: ConformerEnsemble,
    grid: np.ndarray,
    weighted: bool = False,
) -> np.ndarray:
    """
    Main workhorse function for ASO calculation that takes advantage of C++ backend code whenever possible.
    With large grids can be relatively memory intensive, so breaking up the grid into smaller pieces is recommended
    (see "chunky" calculation strategy)
    """
    alldist = molli_xt.cdist32_eu2(ens._coords, grid)
    vdwr2s = np.array([a.vdw_radius for a in ens.atoms]) ** 2
    diff = alldist <= vdwr2s[:, None]

    return np.average(
        np.any(diff, axis=1),
        axis=0,
        weights=ens.weights if weighted else None,
    )
