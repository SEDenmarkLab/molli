import numpy as np
import molli_xt
import molli as ml
from typing import Callable, Any
from .gbca import indicator


def indicator_field(
    ens: ml.ConformerEnsemble,
    grid: np.ndarray,
    indicator_values: np.ndarray,
    atomic_radii: np.ndarray,
    proximal_atom_idx: np.ndarray = None,
    weighted: bool = False,
) -> np.ndarray:
    """
    Main workhorse function for indicator field calculation that takes advantage of C++ backend code whenever possible.
    """
    alldist2 = molli_xt.cdist32_eu2(ens._coords, grid)

    if proximal_atom_idx is None:
        proximal_atom_idx = indicator(grid, ens, max_dist=np.max(atomic_radii))

    assert proximal_atom_idx.shape == (ens.n_conformers, grid.shape[0])

    within_radius = np.any(alldist2 <= (atomic_radii**2)[:, np.newaxis], axis=1)
    filtered_atom_index = np.zeros_like(proximal_atom_idx)

    where = within_radius & (proximal_atom_idx >= 0)

    filtered_atom_index[where] = proximal_atom_idx[where]

    field_all = np.zeros_like(filtered_atom_index, dtype=np.float64)

    for i in range(ens.n_conformers):
        field_all[i, where[i]] = np.take(
            indicator_values[i], filtered_atom_index[i, where[i]]
        )

    return np.average(field_all, axis=0, weights=ens.weights if weighted else None)


def aeif(
    ens: ml.ConformerEnsemble,
    grid: np.ndarray,
    proximal_atom_idx: np.ndarray = None,
    weighted: bool = False,
):
    """`
    Average electronic indicator field
    """
    vdw_radii = np.array([a.vdw_radius for a in ens.atoms])
    charges = ens.atomic_charges
    return indicator_field(
        ens,
        grid,
        charges,
        vdw_radii,
        proximal_atom_idx=proximal_atom_idx,
        weighted=weighted,
    )
