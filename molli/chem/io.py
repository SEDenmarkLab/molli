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
This describes chemical I/O operations for binary serialization
"""

from typing import Literal, Type, IO, List
from . import Molecule, ConformerEnsemble, Atom, Bond
from ..config import VERSION
from enum import IntEnum
import numpy as np
import msgpack
from pathlib import Path

# =====================================
# These schemas were used in the previous iterations of the serialization
# left mostly for backwards compatibility reasons
# These should not be used
# =====================================
ATOM_SCHEMA_V1 = (
    "element",
    "isotope",
    "label",
    "atype",
    "stereo",
    "geom",
)

BOND_SCHEMA_V1 = (
    "a1",
    "a2",
    "label",
    "btype",
    "stereo",
    "f_order",
)

MOLECULE_SCHEMA_V1 = (
    "name",
    "n_atoms",
    "atoms",
    "bonds",
    "charge",
    "mult",
    "coords",
    "atomic_charges",
)

ENSEMBLE_SCHEMA_V1 = (
    "name",
    "n_conformers",
    "n_atoms",
    "atoms",
    "bonds",
    "charge",
    "mult",
    "coords",
    "weights",
    "atomic_charges",
)

# =====================================
# These are the current default schemas
# =====================================

ATOM_SCHEMA_V2 = (
    "element",
    "isotope",
    "label",
    "atype",
    "stereo",
    "geom",
    "formal_charge",
    "formal_spin",
    "attrib",
)

BOND_SCHEMA_V2 = (
    "a1",
    "a2",
    "label",
    "btype",
    "stereo",
    "f_order",
    "attrib",
)

MOLECULE_SCHEMA_V2 = (
    "name",
    "n_atoms",
    "n_bonds",
    "atoms",
    "atomic_charges",
    "bonds",
    "coords",
    "charge",
    "mult",
    "attrib",
)

ENSEMBLE_SCHEMA_V2 = (
    "name",
    "n_conformers",
    "n_atoms",
    "n_bonds",
    "atoms",
    "bonds",
    "charge",
    "mult",
    "coords",
    "weights",
    "atomic_charges",
    "attrib",
)

HIGHEST_VERSION = 2


def _serialize_mol_v1(mol: Molecule):
    atom_id_map = {a: i for i, a in enumerate(mol.atoms)}

    atoms = [a.as_tuple(ATOM_SCHEMA_V1) for a in mol.atoms]
    bonds = [
        (atom_id_map[b.a1], atom_id_map[b.a2]) + b.as_tuple(BOND_SCHEMA_V1[2:])
        for b in mol.bonds
    ]

    return (
        mol.name,
        mol.n_atoms,
        atoms,
        bonds,
        mol.charge,
        mol.mult,
        mol.coords.astype(">f4").tobytes(),
        mol.atomic_charges.astype(">f4").tobytes(),
    )


def _serialize_ens_v1(ens: ConformerEnsemble):
    atom_id_map = {a: i for i, a in enumerate(ens.atoms)}

    atoms = [a.as_tuple(ATOM_SCHEMA_V1) for a in ens.atoms]
    bonds = [
        (atom_id_map[b.a1], atom_id_map[b.a2]) + b.as_tuple(BOND_SCHEMA_V1[2:])
        for b in ens.bonds
    ]

    return (
        ens.name,
        ens.n_conformers,
        ens.n_atoms,
        atoms,
        bonds,
        ens.charge,
        ens.mult,
        ens.coords.astype(">f4").tobytes(),
        ens.weights.astype(">f4").tobytes(),
        ens.atomic_charges.astype(">f4").tobytes(),
    )


def _deserialize_mol_v1(mt: tuple, cls: type[Molecule] = Molecule) -> Molecule:
    # "name",
    # "n_atoms",
    # "atoms",
    # "bonds",
    # "charge",
    # "mult",
    # "coords",
    # "atomic_charges",

    (
        name,
        n_atoms,
        atoms,
        bonds,
        charge,
        mult,
        coords,
        atomic_charges,
    ) = mt

    atoms = [Atom(**dict(zip(ATOM_SCHEMA_V1, a))) for a in atoms]

    #     atoms = []
    #     for i, a in enumerate(_atoms):
    #         atoms.append(Atom(*a))

    res = cls(
        atoms,
        n_atoms=n_atoms,
        name=name,
        charge=charge,
        mult=mult,
        coords=np.frombuffer(coords, dtype=">f4").reshape((n_atoms, 3)),
        atomic_charges=np.frombuffer(atomic_charges, dtype=">f4").reshape((n_atoms)),
    )

    for b in bonds:
        res.connect(*b[:2], **dict(zip(BOND_SCHEMA_V1[2:], b[2:])))

    return res


def _deserialize_ens_v1(
    mt: tuple, cls: type[ConformerEnsemble] = ConformerEnsemble
) -> ConformerEnsemble:
    # "name",
    # "n_conformers",
    # "n_atoms",
    # "atoms",
    # "bonds",
    # "charge",
    # "mult",
    # "coords",
    # "weights",
    # "atomic_charges",

    (
        name,
        n_conformers,
        n_atoms,
        atoms,
        bonds,
        charge,
        mult,
        coords,
        weights,
        atomic_charges,
    ) = mt

    atoms = [Atom(**dict(zip(ATOM_SCHEMA_V1, a))) for a in atoms]

    res = cls(
        atoms,
        n_atoms=n_atoms,
        n_conformers=n_conformers,
        name=name,
        charge=charge,
        mult=mult,
        coords=np.frombuffer(coords, dtype=">f4").reshape((n_conformers, n_atoms, 3)),
        weights=np.frombuffer(weights, dtype=">f4"),
        atomic_charges=np.frombuffer(atomic_charges, dtype=">f4"),
    )

    for b in bonds:
        res.connect(*b[:2], **dict(zip(BOND_SCHEMA_V1[2:], b[2:])))

    return res


def _serialize_mol_v2(mol: Molecule):
    atom_id_map = {a: i for i, a in enumerate(mol.atoms)}

    atoms = [a.as_tuple(ATOM_SCHEMA_V2) for a in mol.atoms]
    bonds = [
        (atom_id_map[b.a1], atom_id_map[b.a2]) + b.as_tuple(BOND_SCHEMA_V2[2:])
        for b in mol.bonds
    ]

    return (
        mol.name,
        mol.n_atoms,
        mol.n_bonds,
        mol.charge,
        mol.mult,
        atoms,
        bonds,
        mol.coords.astype(">f4").tobytes(),
        mol.atomic_charges.astype(">f4").tobytes(),
        mol.attrib,
    )


def _serialize_ens_v2(ens: ConformerEnsemble):
    atom_id_map = {a: i for i, a in enumerate(ens.atoms)}

    atoms = [a.as_tuple(ATOM_SCHEMA_V2) for a in ens.atoms]
    bonds = [
        (atom_id_map[b.a1], atom_id_map[b.a2]) + b.as_tuple(BOND_SCHEMA_V2[2:])
        for b in ens.bonds
    ]

    # "name",
    # "n_conformers",
    # "n_atoms",
    # "n_bonds",
    # "charge",
    # "mult",
    # "atoms",
    # "bonds",
    # "coords",
    # "weights",
    # "atomic_charges",
    # "attrib"

    return (
        ens.name,
        ens.n_conformers,
        ens.n_atoms,
        ens.n_bonds,
        ens.charge,
        ens.mult,
        atoms,
        bonds,
        ens.coords.astype(">f4").tobytes(),
        ens.weights.astype(">f4").tobytes(),
        ens.atomic_charges.astype(">f4").tobytes(),
        ens.attrib,
    )


def _deserialize_mol_v2(mt: tuple, cls: type[Molecule] = Molecule) -> Molecule:
    (
        name,
        n_atoms,
        n_bonds,
        charge,
        mult,
        atoms,
        bonds,
        coords,
        atomic_charges,
        attrib,
    ) = mt

    atoms = [Atom(**dict(zip(ATOM_SCHEMA_V2, a))) for a in atoms]

    #     atoms = []
    #     for i, a in enumerate(_atoms):
    #         atoms.append(Atom(*a))

    res = cls(
        atoms,
        n_atoms=n_atoms,
        name=name,
        charge=charge,
        mult=mult,
        coords=np.frombuffer(coords, dtype=">f4").reshape((n_atoms, 3)),
        atomic_charges=np.frombuffer(atomic_charges, dtype=">f4").reshape((n_atoms)),
        attrib=attrib,
    )

    for b in bonds:
        res.connect(*b[:2], **dict(zip(BOND_SCHEMA_V2[2:], b[2:])))

    return res

    #     res = cls(
    #         atoms,
    #         n_atoms=_na,
    #         name=_name,
    #         charge=_charge,
    #         mult=_mult,
    #         coords=coords,
    #         atomic_charges=atomic_charges,
    #     )

    #     for i, b in enumerate(_bonds):
    #         a1, a2, *b_attr = b
    #         res.append_bond(Bond(atoms[a1], atoms[a2], *b_attr))

    #     return res


def _deserialize_ens_v2(
    mt: tuple, cls: type[ConformerEnsemble] = ConformerEnsemble
) -> ConformerEnsemble:
    # "name",
    # "n_conformers",
    # "n_atoms",
    # "n_bonds",
    # "charge",
    # "mult",
    # "atoms",
    # "bonds",
    # "coords",
    # "weights",
    # "atomic_charges",
    # "attrib"

    (
        name,
        n_conformers,
        n_atoms,
        n_bonds,
        charge,
        mult,
        atoms,
        bonds,
        coords,
        weights,
        atomic_charges,
        attrib,
    ) = mt

    atoms = [Atom(**dict(zip(ATOM_SCHEMA_V2, a))) for a in atoms]

    res = cls(
        atoms,
        n_atoms=n_atoms,
        n_conformers=n_conformers,
        name=name,
        charge=charge,
        mult=mult,
        coords=np.frombuffer(coords, dtype=">f4").reshape((n_conformers, n_atoms, 3)),
        weights=np.frombuffer(weights, dtype=">f4"),
        atomic_charges=np.frombuffer(atomic_charges, dtype=">f4").reshape(
            (n_conformers, n_atoms)
        ),
        attrib=attrib,
    )

    for b in bonds:
        res.connect(*b[:2], **dict(zip(BOND_SCHEMA_V2[2:], b[2:])))

    return res


DESCRIPTOR_MOL_V2 = {
    "molli_version": VERSION,
    "object_class": "Molecule",
    "object_schema": MOLECULE_SCHEMA_V2,
    "atom_schema": ATOM_SCHEMA_V2,
    "bond_schema": BOND_SCHEMA_V2,
}

DESCRIPTOR_ENS_V2 = {
    "molli_version": VERSION,
    "object_class": "ConformerEnsemble",
    "object_schema": ENSEMBLE_SCHEMA_V2,
    "atom_schema": ATOM_SCHEMA_V2,
    "bond_schema": BOND_SCHEMA_V2,
}
