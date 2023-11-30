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
ATOM_SCHEMA_OLD = (
    "element",
    "isotope",
    "label",
    "atype",
    "stereo",
    "geom",
)

BOND_SCHEMA_OLD = (
    "a1",
    "a2",
    "label",
    "btype",
    "stereo",
    "f_order",
)

MOLECULE_SCHEMA_OLD = (
    "name",
    "n_atoms",
    "atoms",
    "bonds",
    "charge",
    "mult",
    "coords",
    "atomic_charges",
)

ENSEMBLE_SCHEMA_OLD = (
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

ATOM_SCHEMA_DEFAULT = (
    "element",
    "isotope",
    "label",
    "atype",
    "stereo",
    "geom",
    "charge",
    "spin",
    "attrib",
)

BOND_SCHEMA_DEFAULT = (
    "a1",
    "a2",
    "label",
    "btype",
    "stereo",
    "f_order",
    "attrib",
)

MOLECULE_SCHEMA_DEFAULT = (
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

ENSEMBLE_SCHEMA_DEFAULT = (
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
    "attrib",
)


def _serialize_mb(mol: Molecule | ConformerEnsemble, schema: List[str] = None):
    """
    This provides a binary serialization in a form of a messagepack dictionary with the outer schema defined
    """
    ...


def _deserialize_mb(
    cls: Type[Molecule] | Type[ConformerEnsemble], data: tuple, schema: List[str] = None
):
    """
    This provides a deserialization of messagepack decoded tuples
    """
    # This is to ensure backwards compatibility with the previous mlib data storage
    if schema is None:
        if len(data) == 8:
            schema = (
                "name",
                "n_atoms",
                "atoms",
                "bonds",
                "charge",
                "mult",
                "coords",
                "atomic_charges",
            )
        elif len(data) == 10:
            schema = (
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

    # This is a dictionary that can initialize molecule or conformer objects
    mol_dict = dict(zip(schema, data))

    pass


def load(
    stream: IO,
    format: str,
    parser: Literal["molli", "openbabel", "obabel", None] = None,
    type: Literal["molecule", "ensemble", None] | Type = None,
):
    """This is a universal loader of molecules / ensembles"""
    pass
