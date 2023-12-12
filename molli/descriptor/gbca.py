"""
Grid based conformer averaged descriptors
This is a foundational file
"""
import numpy as np
from numpy.typing import ArrayLike
import molli as ml


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


def indicator(
    grid: np.ndarray,
    struct_or_ens: ml.ConformerEnsemble | ml.CartesianGeometry,
    dtype: np.dtype = np.int64,
    max_dist: float | str = "vdw_radius",
):
    """Returns an indicator array
    in the shape of (n_conformers, n_gridpoints) for ensembles
    and in the shape of (n_gridpoints,) for StructureLikes
    where the number corresponds to the *atom index* (or -1 if the closest atom is farther than max_dist)

    Uses KDTree data structure to increase the rate of computations significantly
    """
    from scipy.spatial import KDTree

    # This is to handle very specific cases
    if isinstance(max_dist, str):
        max_dist = np.array([getattr(a, max_dist) for a in struct_or_ens.atoms])

    if isinstance(struct_or_ens, ml.ConformerEnsemble):
        result = np.empty((struct_or_ens.n_conformers, grid.shape[0]), dtype=dtype)

        for i, c in enumerate(struct_or_ens):
            ckd = KDTree(c.coords)
            dd, ii = ckd.query(grid, distance_upper_bound=max_dist, k=1)
            result[i] = np.where(dd <= max_dist, ii, -1)

    elif isinstance(struct_or_ens, ml.CartesianGeometry):
        dd, ii = KDTree(struct_or_ens.coords).query(grid, distance_upper_bound=2.0, k=1)
        result = np.asarray(np.where(dd <= 2.0, ii, -1), dtype=dtype)

    return result


def prune(
    grid: np.ndarray,
    struct_or_ens: ml.ConformerEnsemble | ml.CartesianGeometry,
    eps: float = 0.5,
    max_dist: float = 2.0,
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
    return np.where(dd <= max_dist)[0]
