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
    __slots__ = Structure.__slots__ + ("_atomic_charges",)

    def __init__(
        self,
        other: Structure | PromoleculeLike = None,
        /,
        n_atoms: int = 0,
        *,
        charge: int = None,
        mult: int = None,
        name: str = None,
        atomic_charges: ArrayLike = ...,
        **kwds,
    ):
        """
        If other is not none, that molecule will be cloned.
        """
        # if isinstance(other, Molecule | Structure):
        #     ...
        super().__init__(
            other, n_atoms=n_atoms, name=name, charge=charge, mult=mult, **kwds
        )
        self.atomic_charges = atomic_charges

    # @property
    # def charge(self):
    #     return self._charge

    # @charge.setter
    # def charge(self, other: int):
    #     self._charge = other

    @property
    def atomic_charges(self) -> np.ndarray:
        """
        The atomic charges of the molecule.
        
        Returns:
            np.ndarray: The atomic charges of the molecule.
        
        Example Usage:
            >>> mol = Molecule(H20)
            >>> print(mol.atomic_charges) # [1,1,-2]
        """        
        return self._atomic_charges

    @atomic_charges.setter
    def atomic_charges(self, other: ArrayLike):
        """
        Sets the atomic charges of the molecule.

        Args:
            other (ArrayLike): The atomic charges of the molecule.
        
        Raises:
            ValueError: Inappropriate shape of atomic charge array
        """        
        if other is Ellipsis:
            self._atomic_charges = np.zeros(self.n_atoms)
        else:
            _pc = np.array(other, self._coords_dtype)
            if _pc.shape == (self.n_atoms,):
                self._atomic_charges = _pc
            else:
                raise ValueError("Inappropriate shape of atomic charge array")

    def dump_mol2(self, stream: StringIO = None):
        if stream is None:
            stream = StringIO()
            
        stream.write(f"# Produced with molli package\n")
        stream.write(
            f"@<TRIPOS>MOLECULE\n{self.name}\n{self.n_atoms} {self.n_bonds} 0 0 0\nSMALL\nUSER_CHARGES\n\n"
        )

        stream.write("@<TRIPOS>ATOM\n")
        for i, a in enumerate(self.atoms):
            x, y, z = self.coords[i]
            c = 0.0 #Currently needs to be updated to be inherited within the structure or even individual atoms
            label = a.label or a.element.symbol
            atype = a.get_mol2_type() or a.element.symbol
            stream.write(
                f"{i+1:>6} {label:<3} {x:>12.6f} {y:>12.6f} {z:>12.6f} {atype:<10} 1 UNL1 {c}\n"
            )

        stream.write("@<TRIPOS>BOND\n")
        for i, b in enumerate(self.bonds):
            a1, a2 = self.atoms.index(b.a1), self.atoms.index(b.a2)
            btype = b.get_mol2_type()            
            stream.write(f"{i+1:>6} {a1+1:>6} {a2+1:>6} {btype:>10}\n")
        
    def dumps_mol2(self) -> str:
        """
        Returns a mol2 file as a string.
        
        Returns:
            str: The mol2 file as a string.
        """        
        stream = StringIO()
        self.dump_mol2(stream)        
        return stream.getvalue()

StructureLike = Molecule | Structure | Substructure

