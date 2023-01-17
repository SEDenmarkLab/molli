from __future__ import annotations
from typing import Any, List, Iterable, Generator, TypeVar, Generic, Dict
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from io import StringIO, BytesIO
import re
from warnings import warn

from . import (
    Element,
    PromoleculeLike,
    Atom,
    AtomLike,
    Bond,
    Connectivity,
    CartesianGeometry,
    Structure,
    Substructure,
)


class Molecule(Structure):
    """Fundamental class of the MOLLI Package."""

    __slots__ = ("_charge", "multiplicity", "_partial_charges") + Structure.__slots__

    def __init__(
        self,
        other: Structure | PromoleculeLike = None,
        /,
        n_atoms: int = 0,
        *,
        charge: int = 0,
        multiplicity: int = 1,
        name: str = None,
        atomic_charges: ArrayLike = ...,
        **kwds,
    ):
        """
        If other is not none, that molecule will be cloned.
        """
        # if isinstance(other, Molecule | Structure):
        #     ...
        super().__init__(other, n_atoms=n_atoms, name=name)
        self.charge = charge
        self.multiplicity = multiplicity
        self.atomic_charges = atomic_charges

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, other: int):
        self._charge = other

    @property
    def atomic_charges(self):
        return self._atomic_charges

    @atomic_charges.setter
    def atomic_charges(self, other: ArrayLike):
        if other is Ellipsis:
            self._atomic_charges = np.zeros(self.n_atoms)
        else:
            _pc = np.array(other, self._coords_dtype)
            if _pc.shape == (self.n_atoms,):
                self._atomic_charges = _pc
            else:
                raise ValueError("Inappropriate shape of atomic charge array")

StructureLike = Molecule | Structure | Substructure

