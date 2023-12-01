# This describes chemical I/O operations
from typing import Literal, Type, IO, List
from .molecule import Molecule
from .ensemble import ConformerEnsemble
from enum import IntEnum
import numpy as np
import msgpack

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
    # "a1",
    # "a2",
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
    # "a1",
    # "a2",
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
        (atom_id_map[b.a1], atom_id_map[b.a2]) + b.as_tuple(BOND_SCHEMA_V1)
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
        (atom_id_map[b.a1], atom_id_map[b.a2]) + b.as_tuple(BOND_SCHEMA_V1)
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


def _serialize_mol_v2(mol: Molecule):
    atom_id_map = {a: i for i, a in enumerate(mol.atoms)}

    atoms = [a.as_tuple(ATOM_SCHEMA_V2) for a in mol.atoms]
    bonds = [
        (atom_id_map[b.a1], atom_id_map[b.a2]) + b.as_tuple(BOND_SCHEMA_V2)
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
        (atom_id_map[b.a1], atom_id_map[b.a2]) + b.as_tuple(BOND_SCHEMA_V2)
        for b in ens.bonds
    ]

    return (
        ens.name,
        ens.n_conformers,
        ens.n_atoms,
        ens.charge,
        ens.mult,
        atoms,
        bonds,
        ens.coords.astype(">f4").tobytes(),
        ens.weights.astype(">f4").tobytes(),
        ens.atomic_charges.astype(">f4").tobytes(),
        ens.attrib,
    )


def load(
    stream: IO,
    format: str,
    parser: Literal["molli", "openbabel", "obabel", None] = None,
    type: Literal["molecule", "ensemble", None] | Type = None,
):
    """This is a universal loader of molecules / ensembles"""
    pass
