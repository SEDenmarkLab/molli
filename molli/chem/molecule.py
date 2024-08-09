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
# `molli.chem.molecule`
This submodule defines the most crucial part of molli: the `Molecule` class
"""

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
    """This is the fundamental class of the Molli package. This class
    inherits different methods from Promolecule, Connectivity,
    CartesianGeometry, and Structure.
    """

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
    def atomic_charges(self) -> np.ndarray:
        """The atomic charges of the Molecule in shape (n_atoms,)

        Returns
        -------
        np.ndarray
            Returns the atomic charges of the Molecule

        Examples
        -------
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.atomic_charges
            array([-0.2981,  0.0024, ...
        """

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

    def dump_mol2(self, stream: StringIO) -> None:
        """Dumps the mol2 block into the output stream

        Parameters
        ----------
        stream : StringIO, optional
            Output stream, by default None

        Examples
        -------
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> with open('dendrobine.mol2', 'w') as f:
            >>>     dendrobine.dump_mol2(f)
            # Produced with molli package
            @<TRIPOS>MOLECULE
            dendrobine
            ...
        """

        try:
            _name = self.name
        except:
            _name = "*****"

        stream.write(f"# Produced with molli package\n")
        stream.write(
            f"@<TRIPOS>MOLECULE\n{_name}\n{self.n_atoms} {self.n_bonds} 0 0"
            " 0\nSMALL\nUSER_CHARGES\n\n"
        )

        stream.write("@<TRIPOS>ATOM\n")
        for i, a in enumerate(self.atoms):
            x, y, z = self.coords[i]
            c = self.atomic_charges[i] or 0.0
            label = a.label or a.element.symbol
            atype = a.get_mol2_type() or a.element.symbol
            stream.write(
                f"{i+1:>6} {label:<3} {x:>12.6f} {y:>12.6f} {z:>12.6f} {atype:<10} 1 UNL1 {c:0.3f}\n"
            )

        stream.write("@<TRIPOS>BOND\n")
        for i, b in enumerate(self.bonds):
            a1, a2 = self.atoms.index(b.a1), self.atoms.index(b.a2)
            btype = b.get_mol2_type()
            stream.write(f"{i+1:>6} {a1+1:>6} {a2+1:>6} {btype:>3}\n")

    def dumps_mol2(self) -> str:
        """Dumps the mol2 block as a string

        Returns
        -------
        str
            The mol2 block

        Examples
        -------
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.dumps_mol2()
            # Produced with molli package
            @<TRIPOS>MOLECULE
            dendrobine
            ...
        """

        with StringIO() as stream:
            self.dump_mol2(stream)
            return stream.getvalue()

    def add_atom(self, a: Atom, coord: ArrayLike, charge: float = None) -> None:
        """Adds atom to Molecule

        Parameters
        ----------
        a : Atom
            Atom to add
        coord : ArrayLike
            Coordinates to add
        charge : float, optional
            Charge of atom, by default None

        Examples
        -------
        The Molecule class inherits add_atom()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.n_atoms
            44
            >>> new_atom = ml.Atom("C")
            >>> dendrobine.add_atom(new_atom, [0,0,0])
            >>> dendrobine.n_atoms
            45
        """
        super().add_atom(a, coord)
        self._atomic_charges = np.append(self._atomic_charges, [charge], axis=0)

    def del_atom(self, _a: AtomLike) -> None:
        """Deletes an atom from the Molecule

        Parameters
        ----------
        _a : AtomLike
            An atom, index, label, or Element. This will only delete the first
            instance of the label or Element found

        Examples
        -------
        The Molecule class inherits del_atom()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.get_atom(0)
            Atom(element=N, isotope=None, label='N', formal_charge=0, formal_spin=0)
            >>> dendrobine.del_atom(0)
            >>> dendrobine.get_atom(0)
            Atom(element=C, isotope=None, label='C', formal_charge=0, formal_spin=0)
        """
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
"""
StructureLike can be a Molecule, Structure, or Substructure
"""
