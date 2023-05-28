from ..chem import Molecule, ConformerEnsemble
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
from math import dist
import molli_xt


def _where_distance(p, g: np.ndarray, r: float) -> np.ndarray:
    """
    Return a numpy array of where point is within radius from gridpoints
    """

    diff = g - p
    dists = np.sum(diff**2, axis=-1)

    return dists <= r**2


def aso1(ens: ConformerEnsemble, g: np.ndarray, dtype: str = "float32") -> np.ndarray:
    aso_full = np.empty((ens.n_conformers, g.shape[0]), dtype=dtype)

    # Iterate over conformers in the ensemble
    # Complexity (O(n_confs * n_gpts))
    for i, c in enumerate(ens):
        # Iterate over atoms
        for j, a in enumerate(c.atoms):
            aso_full[i][_where_distance(c.coords[j], g, a.vdw_radius)] = 1

    aso = np.average(aso_full, axis=0)
    return aso


def aso2(ens: ConformerEnsemble, g: np.ndarray, dtype: str = "float32") -> np.ndarray:
    alldist = molli_xt.cdist32_eu2_f3(ens._coords, g)
    vdwr2s = np.array([a.vdw_radius for a in ens.atoms]) ** 2
    diff = alldist <= vdwr2s[:, None]

    return np.average(np.any(diff, axis=1), axis=0)

def chunky_aso(ens: ConformerEnsemble, grid: np.ndarray, chunksize=512) -> np.ndarray:
    aso_chunks = []
    for subgrid in np.array_split(grid, grid.shape[0] // chunksize):
        aso_chunks.append(aso2(ens, subgrid))
    return np.concatenate(aso_chunks)

def parallel_aso(ens: ConformerEnsemble, grid: np.ndarray, n_threads=4, chunksize=512) -> np.ndarray:
    with ThreadPoolExecutor(n_threads) as tp:
        futures: list[Future] = []
        for subgrid in np.array_split(grid, grid.shape[0] // chunksize):
            f = tp.submit(aso2, ens, subgrid)
            futures.append(f)
        
        sub_asos = [f.result() for f in futures]
    
    return np.concatenate(sub_asos)


# def aso3(ens: ConformerEnsemble, g: np.ndarray, dtype: str = "float32") -> np.ndarray:
#     aso_full = np.empty((ens.n_conformers, g.shape[0]), dtype=dtype)

#     vdw2 = np.array([a.vdw_radius**2 for a in ens.atoms], dtype=dtype)

#     for i, c in enumerate(ens):
#         vecs = g[:, np.newaxis] - c.coords  # Shape of vecs: (n_gpts, n_atoms)
#         dists2 = np.sum(vecs**2, axis=-1)  # Squared distances of grid points to atoms
#         where = np.any(dists2 - vdw2 <= 0, axis=-1)

#         aso_full[i][where] = 1

#     return np.average(aso_full, 0)
