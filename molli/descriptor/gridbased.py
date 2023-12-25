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
# `molli.descriptor.gridbased`

Grid based conformer averaged descriptors
This is a foundational file
"""

import numpy as np
from numpy.typing import ArrayLike
import molli as ml
import numpy as np
import molli_xt
from typing import Callable, Any
import deprecated

__all__ = [
    "rectangular_grid",
    "prune",
    "nearest_atom_index",
    "atomic_indicator_field",
    "aso",
    "aeif",
]


def rectangular_grid(
    r1: ArrayLike,
    r2: ArrayLike,
    padding: float = 0.0,
    spacing: float = 1.0,
    dtype: str = "float32",
) -> np.ndarray:
    l = np.array(r1, dtype=dtype) - padding
    r = np.array(r2, dtype=dtype) + padding

    # Number of points
    nx = int((r[0] - l[0]) // spacing) + 1
    ny = int((r[1] - l[1]) // spacing) + 1
    nz = int((r[2] - l[2]) // spacing) + 1

    # Offsets
    ox = (r[0] - l[0] - (nx - 1) * spacing) / 2
    oy = (r[1] - l[1] - (ny - 1) * spacing) / 2
    oz = (r[2] - l[2] - (nz - 1) * spacing) / 2

    # X linsp
    xs = np.linspace(l[0] + ox, r[0] - ox, nx, endpoint=True, dtype=dtype)
    ys = np.linspace(l[1] + oy, r[1] - oy, ny, endpoint=True, dtype=dtype)
    zs = np.linspace(l[2] + oz, r[2] - oz, nz, endpoint=True, dtype=dtype)

    xx, yy, zz = np.meshgrid(xs, ys, zs)

    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


def nearest_atom_index(
    grid: np.ndarray,
    struct_or_ens: ml.ConformerEnsemble | ml.CartesianGeometry,
    max_dist: float = 2.0,
):
    """Returns an array of atom indexes that are closest to the grid points
    in the shape of (n_conformers, n_gridpoints) for ensembles
    and in the shape of (n_gridpoints,) for StructureLikes
    where the number corresponds to the *atom index* (or -1 if the closest atom is farther than max_dist)

    Uses KDTree data structure to increase the rate of computations significantly
    """
    from scipy.spatial import KDTree

    if isinstance(struct_or_ens, ml.ConformerEnsemble):
        result = np.empty((struct_or_ens.n_conformers, grid.shape[0]), dtype=np.int64)

        for i, c in enumerate(struct_or_ens):
            ckd = KDTree(c.coords)
            dd, ii = ckd.query(grid, distance_upper_bound=max_dist, k=1)
            result[i] = np.where(dd <= max_dist, ii, -1)

    elif isinstance(struct_or_ens, ml.CartesianGeometry):
        dd, ii = KDTree(struct_or_ens.coords).query(grid, distance_upper_bound=2.0, k=1)
        result = np.where(dd <= 2.0, ii, -1)

    return result


def prune(
    grid: np.ndarray,
    struct_or_ens: ml.ConformerEnsemble | ml.CartesianGeometry,
    max_dist: float = 2.0,
    eps: float = 0.5,
):
    """
    This function prunes the grid to return only the points closer than (max_dist + eps) from an ensemble or molecule.
    This is a very fast (!) implementation that utilizes a KDTree structure for fast queries
    """
    from scipy.spatial import KDTree

    if isinstance(struct_or_ens, ml.ConformerEnsemble):
        kdt = KDTree(np.vstack(struct_or_ens.coords))
    else:
        kdt = KDTree(struct_or_ens.coords)
    dd, ii = kdt.query(grid, eps=eps, distance_upper_bound=max_dist)
    return np.asarray(np.where(dd <= max_dist)[0])


def atomic_indicator_field(
    ens: ml.ConformerEnsemble,
    grid: np.ndarray,
    indicator_values: np.ndarray,
    atomic_radii: np.ndarray,
    nearest_atom_idx: np.ndarray = None,
    weighted: bool = False,
) -> np.ndarray:
    """
    Main workhorse function for indicator field calculation that takes advantage of C++ backend code whenever possible.
    """
    alldist2 = molli_xt.cdist32_eu2(ens._coords, grid)

    if nearest_atom_idx is None:
        nearest_atom_idx = nearest_atom_index(grid, ens, max_dist=np.max(atomic_radii))

    assert nearest_atom_idx.shape == (ens.n_conformers, grid.shape[0])

    within_radius = np.any(alldist2 <= (atomic_radii**2)[:, np.newaxis], axis=1)
    filtered_atom_index = np.zeros_like(nearest_atom_idx)

    where = within_radius & (nearest_atom_idx >= 0)

    filtered_atom_index[where] = nearest_atom_idx[where]

    field_all = np.zeros_like(filtered_atom_index, dtype=np.float64)

    for i in range(ens.n_conformers):
        field_all[i, where[i]] = np.take(
            indicator_values[i], filtered_atom_index[i, where[i]]
        )

    return np.average(field_all, axis=0, weights=ens.weights if weighted else None)


def aeif(
    ens: ml.ConformerEnsemble,
    grid: np.ndarray,
    nearest_atom_idx: np.ndarray = None,
    weighted: bool = False,
):
    """`
    Average electronic indicator field
    """
    vdw_radii = np.array([a.vdw_radius for a in ens.atoms])
    charges = ens.atomic_charges
    return atomic_indicator_field(
        ens,
        grid,
        charges,
        vdw_radii,
        nearest_atom_idx=nearest_atom_idx,
        weighted=weighted,
    )


def aso(
    ens: ml.ConformerEnsemble,
    grid: np.ndarray,
    weighted: bool = False,
) -> np.ndarray:
    """
    Main workhorse function for ASO calculation that takes advantage of C++ backend code whenever possible.
    With large grids can be relatively memory intensive, so breaking up the grid into smaller pieces is recommended
    (see "chunky" calculation strategy)
    """
    alldist = molli_xt.cdist32f_eu2(ens._coords, grid)
    vdwr2s = np.array([a.vdw_radius for a in ens.atoms]) ** 2
    diff = alldist <= vdwr2s[:, None]

    return np.average(
        np.any(diff, axis=1),
        axis=0,
        weights=ens.weights if weighted else None,
    )


## Below are mostly legacy definitions


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
    ens: ml.ConformerEnsemble,
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
