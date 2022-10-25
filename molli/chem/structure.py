from __future__ import annotations
from typing import Any, List, Iterable, Generator, TypeVar, Generic
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from . import Atom, AtomLike, Bond, Connectivity, CartesianGeometry, Promolecule, PromoleculeLike, Element
from ..parsing import read_mol2
import re
from io import StringIO, BytesIO
from warnings import warn

# The intent of this regex is that all molecule names must be valid variable and file names.
# This may be useful later.
RE_MOL_NAME = re.compile(r"[_a-zA-Z0-9]+")
RE_MOL_ILLEGAL = re.compile(r"[^_a-zA-Z0-9]")


class Structure(CartesianGeometry, Connectivity):
    """Structure is a simple amalgamation of the concepts of CartesianGeometry and Connectivity"""

    # Improve efficiency of attribute access
    __slots__ = ("_name", "_dtype", "_atoms", "_bonds", "_coords")

    def __init__(
        self,
        n_atoms: int = 0,
        name: str = "unnamed",
        *,
        dtype: str = "float32",
    ):
        """Structure."""
        super().__init__(n_atoms=n_atoms, dtype=dtype)
        self.name = name

    # @property
    # def name(self):
    #     return self._name

    # @name.setter
    # def name(self, value: str):
    #     if RE_MOL_NAME.fullmatch(value):
    #         self._name = value
    #     else:
    #         sub = RE_MOL_ILLEGAL.sub("_", value)
    #         self._name = sub
    #         warn(f"Replaced illegal characters in molecule name: {value} --> {sub}")

    #         # raise ValueError(
    #         #     f"Inappropriate name for a molecule: {value}. Must match {RE_MOL_NAME.pattern}"
    #         # )

    @classmethod
    def yield_from_xyz(
        cls,
        input: str | StringIO,
        name: str = "unnamed",
        source_units: str = "Angstrom",
    ):
        for g in super().yield_from_xyz(input, source_units):
            g.name = name
            yield g

    @classmethod
    def from_xyz(
        cls,
        input: str | StringIO,
        name: str = "unnamed",
        source_units: str = "Angstrom",
    ):
        return next(cls.yield_from_xyz(input, name, source_units))
    
    @classmethod
    def yield_from_mol2(
        cls: type[Structure],
        input: str | StringIO,
        name: str = ...,
        dtype: str = "float32",
    ):
        mol2io = StringIO(input) if isinstance(input, str) else input

        for block in read_mol2(mol2io):
            _name = name if isinstance(name, str) else block.header.name 
            res = cls(n_atoms=block.header.n_atoms, name=_name, dtype=dtype)

            for i, a in enumerate(block.atoms):
                res.atoms[i].label = a.label
                res.atoms[i].element = Element[a.element]
                # res.atoms[i].meta.mol2_type = a.typ
                res.coords[i] = (a.x, a.y, a.z)
            
            for i, b in enumerate(block.bonds):
                Bond.add_to(res, res.atoms[b.a1 - 1], res.atoms[b.a2 - 1], b.order)
            
            yield res
    
    @classmethod
    def from_mol2(
        cls: type[Structure],
        input: str | StringIO,
        name: str = ...,
        dtype: str = "float32",
    ):
        return next(cls.yield_from_mol2(input=input, name=name, dtype=dtype))
    
    def to_dict(self):
        bonds = []


        res = {
            "name": self.name,
            "n_atoms": self.n_atoms,
            "n_bonds": self.n_bonds,
            "dtype": self._dtype,
            "atoms": [a.to_dict() for a in self.atoms],
            "bonds": [b.to_dict() for b in self.bonds],
            "coords": self._coords.flatten().tolist()
        }

        return res
    
    @classmethod
    def from_dict(self):
        ...


    def substructure(self, atoms: Iterable[AtomLike]) -> Substructure:
        return Substructure(self, list(atoms))
    
    @property
    def heavy(self) -> Substructure:
        return Substructure(self, [a for a in self.atoms if a.element != Element.H])

    def bond_length(self, b: Bond) -> float:
        return self.distance(b.a1, b.a2)
    
    def bond_vector(self, b: Bond) -> np.ndarray:
        i1, i2 = self.yield_atom_indices((b.a1, b.a2))
        return self.coords[i2] - self.coords[i1]
    
    def bond_coords(self, b: Bond) -> tuple[np.ndarray]:
        return self.coord_subset((b.a1, b.a2))

class Substructure(Structure):
    def __init__(self, parent: Structure, atoms: List[AtomLike]):
        self._parent = parent
        self._atoms = atoms
        self._bonds = []

        for b in parent.bonds:
            if b.a1 in self.atoms and b.a2 in self.atoms:
                self._bonds.append(b)
    
    def yield_parent_atom_indices(self, atoms: Iterable[AtomLike]) -> Generator[int, None, None]:
        yield from self._parent.yield_atom_indices(atoms)
    
    @property
    def parent_atom_indices(self):
        return list(self._parent.yield_atom_indices(self._atoms))
    
    @property
    def coords(self):
        return self._parent.coords[self.parent_atom_indices]
    
    @coords.setter
    def coords(self, other):
        self._parent.coords[self.parent_atom_indices] = other



    
    
    