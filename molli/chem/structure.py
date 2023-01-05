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
from itertools import chain

# The intent of this regex is that all molecule names must be valid variable and file names.
# This may be useful later.
RE_MOL_NAME = re.compile(r"[_a-zA-Z0-9]+")
RE_MOL_ILLEGAL = re.compile(r"[^_a-zA-Z0-9]")


class Structure(CartesianGeometry, Connectivity):
    """Structure is a simple amalgamation of the concepts of CartesianGeometry and Connectivity"""

    # Improve efficiency of attribute access
    __slots__ = ("_name", "_atoms", "_bonds", "_coords")

    def __init__(
        self,
        other: Structure = None,
        /,
        *,
        n_atoms: int = 0,
        name: str = "unnamed",
        coords: ArrayLike = None,
        copy_atoms: bool = False,
        **kwds,
    ):
        """Structure."""
        super().__init__(other, n_atoms=n_atoms, coords=coords, name=name, copy_atoms=copy_atoms, **kwds)

   
    @classmethod
    def yield_from_mol2(
        cls: type[Structure],
        input: str | StringIO,
        name: str = "unnamed",
    ):
        mol2io = StringIO(input) if isinstance(input, str) else input

        for block in read_mol2(mol2io):
            _name = name or block.header.name 
            res = cls(None, n_atoms=block.header.n_atoms, name=_name)

            for i, a in enumerate(block.atoms):
                try:
                    res.atoms[i].element = Element.get(a.element)
                except:
                    res.atoms[i].element = Element.Unknown

                res.coords[i] = a.xyz
                res.atoms[i].mol2_type = a.typ
            
            for i, b in enumerate(block.bonds):
                res.append_bond(Bond(res.atoms[b.a1 - 1], res.atoms[b.a2 - 1], mol2_type=b.typ))
            
            yield res
    
    @classmethod
    def from_mol2(
        cls: type[Structure],
        input: str | StringIO,
        name: str = None,
    ):
        return next(cls.yield_from_mol2(input=input, name=name))
    
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
    
    @classmethod
    def concatenate(cls: type[Structure], *structs: Structure) -> Structure:

        source_atoms = list(chain.from_iterable(x.atoms for x in structs))
        res = cls(source_atoms, copy_atoms=True)

        atom_map = {source_atoms[i] : res.atoms[i] for i in range(res.n_atoms) }
        
        for j, b in enumerate(chain.from_iterable(x.bonds for x in structs)):
            res.append_bond(Bond.evolve(b, a1 = atom_map[b.a1], a2 = atom_map[b.a2]))
        
        res.coords = np.vstack([x.coords for x in structs])

        return res


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
    
    def __or__(self, other: Structure) -> Structure:
        return Structure.concatenate(self, other)
        

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



    
    
    