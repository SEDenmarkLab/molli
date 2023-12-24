from __future__ import annotations
from ast import Tuple
from typing import Any, Callable, List, Iterable, Generator, TypeVar, Generic, Dict
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from io import StringIO, BytesIO
import re
from warnings import warn

from molli.chem import Atom, AtomLike

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

    def __init__(
        self,
        other: Structure | PromoleculeLike = None,
        /,
        n_atoms: int = 0,
        *,
        charge: int = None,
        mult: int = None,
        name: str = None,
        coords: ArrayLike = None,
        atomic_charges: ArrayLike = ...,
        **kwds,
    ):
        """
        If other is not none, that molecule will be cloned.
        """
        # if isinstance(other, Molecule | Structure):
        #     ...
        super().__init__(
            other,
            n_atoms=n_atoms,
            name=name,
            charge=charge,
            mult=mult,
            coords=coords,
            **kwds,
        )
        self.atomic_charges = atomic_charges

    # @property
    # def charge(self):
    #     return self._charge

    # @charge.setter
    # def charge(self, other: int):
    #     self._charge = other

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
                raise ValueError(
                    f"Inappropriate shape of atomic charge array. Received: {_pc.shape}, expected:"
                    f" {(self.n_atoms,)}"
                )

    def dump_mol2(self, stream: StringIO = None):
        if stream is None:
            stream = StringIO()

        stream.write(f"# Produced with molli package\n")
        stream.write(
            f"@<TRIPOS>MOLECULE\n{self.name}\n{self.n_atoms} {self.n_bonds} 0 0"
            " 0\nSMALL\nUSER_CHARGES\n\n"
        )

        stream.write("@<TRIPOS>ATOM\n")
        for i, a in enumerate(self.atoms):
            x, y, z = self.coords[i]
            c = 0.0  # Currently needs to be updated to be inherited within the structure or even individual atoms
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
        This returns a mol2 file as a string
        """
        stream = StringIO()
        self.dump_mol2(stream)
        return stream.getvalue()

    def add_atom(self, a: Atom, coord: ArrayLike, charge: float = None):
        super().add_atom(a, coord)
        self._atomic_charges = np.append(self._atomic_charges, [charge], axis=0)

    def del_atom(self, _a: AtomLike):
        _i = self.get_atom_index(_a)
        super().del_atom(_a)
        self._atomic_charges = np.delete(self._atomic_charges, _i, axis=0)

    def align_to_ref_coords(
        self,
        func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]],
        substructure_indices: list[list[int]],
        reference_subgeometry: Substructure,
        vec: list = None,
    ) -> float:
        centroid = self.substructure(substructure_indices[0]).centroid()
        self.translate(-centroid)

        smallest_rmsd = 100.0
        optimal_rot_matrix = None

        for idx in substructure_indices:
            subgeom = self.substructure(idx)

            rotation, rmsd_ = func(subgeom.coords, reference_subgeometry.coords)

            if rmsd_ < smallest_rmsd:
                smallest_rmsd = rmsd_
                optimal_rot_matrix = rotation

        self.transform(optimal_rot_matrix)

        # bringing ensemble coordinates from origin back to referential coordinates (if necessary)
        if vec is not None:
            self.translate(vec)
            # for cf in self:
            #     cf.coords += vec

        return smallest_rmsd

    # def serialize(self):
    #     atom_id_map = {a: i for i, a in enumerate(self.atoms)}

    #     atoms = [a.as_tuple() for a in self.atoms]
    #     bonds = [b.as_tuple(atom_id_map=atom_id_map) for b in self.bonds]

    #     return (
    #         self.name,
    #         self.n_atoms,
    #         atoms,
    #         bonds,
    #         self.charge,
    #         self.mult,
    #         self.coords.astype(">f4").tobytes(),
    #         self.atomic_charges.astype(">f4").tobytes(),
    #         self.attrib,
    #     )

    # @classmethod
    # def deserialize(cls, struct: tuple):
    #     (
    #         _name,
    #         _na,
    #         _atoms,
    #         _bonds,
    #         _charge,
    #         _mult,
    #         _coords,
    #         _atomic_charges,
    #         _attrib,
    #     ) = struct

    #     coords = np.frombuffer(_coords, dtype=">f4").reshape((_na, 3))
    #     atomic_charges = np.frombuffer(_atomic_charges, dtype=">f4")

    #     atoms = []
    #     for i, a in enumerate(_atoms):
    #         atoms.append(Atom(*a))

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


StructureLike = Molecule | Structure | Substructure
