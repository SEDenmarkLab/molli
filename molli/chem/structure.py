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
This submodule defines the cornerstone of the diamond inheritance in `molli`: 
the `Structure` and `Substructure` classes
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, List, Iterable, Generator, TypeVar, Generic, IO, Callable
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from . import (
    Atom,
    AtomLike,
    AtomType,
    AtomGeom,
    Bond,
    BondType,
    BondStereo,
    Connectivity,
    CartesianGeometry,
    Promolecule,
    PromoleculeLike,
    Element,
    DistanceUnit,
)
from ..math import (
    rotation_matrix_from_vectors,
    _optimize_rotation,
    rotation_matrix_from_axis,
)
from ..parsing import read_mol2
import re
from io import StringIO, BytesIO
from warnings import warn
from itertools import chain
from math import ceil, floor

# The intent of this regex is that all molecule names must be valid variable and file names.
# This may be useful later.
RE_MOL_NAME = re.compile(r"[_a-zA-Z0-9]+")
RE_MOL_ILLEGAL = re.compile(r"[^_a-zA-Z0-9]")


class Structure(CartesianGeometry, Connectivity):
    """Combines the functionality of `CartesianGeometry` andd `Connectivity`
    'CartesianGeometry' gives the molecular data structure features of a 3d
    coordinate matrix
    'Connectivity' gives the molecular data structure features of an
    undirected graph
    """

    def __init__(
        self,
        other: Structure = None,
        /,
        *,
        n_atoms: int = 0,
        name: str = None,
        coords: ArrayLike = None,
        copy_atoms: bool = False,
        charge: int = None,
        mult: int = None,
        **kwds,
    ):
        super().__init__(
            other,
            n_atoms=n_atoms,
            coords=coords,
            name=name,
            copy_atoms=copy_atoms,
            charge=charge,
            mult=mult,
            **kwds,
        )

    @classmethod
    def yield_from_mol2(
        cls: type[Structure],
        input: str | StringIO,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> Generator[Structure, None, None]:
        """Yields generator of Structure from stream

        Parameters
        ----------
        cls : type[Structure]
            The class to load the mol2 file into
        input : str | StringIO
            Stream to read from
        name : str, optional
            Name of the Structure, by default None
        source_units : str, optional
            Units to use when reading, by default "Angstrom"

        Yields
        ------
        Generator[Structure, None, None]
            Yields generator of Structure

        Examples
        -------
        The Molecule class inherits yield_from_mol2()
            >>> with open(ml.files.dendrobine_mol2) as f:
            >>>     ml.Molecule.yield_from_mol2(f, name='dendrobine')
            <generator object Structure.yield_from_mol2 at ...>
        If desired, one can work directly with Structure class instead
            >>> with open(ml.files.dendrobine_mol2) as f:
            >>>     ml.Structure.yield_from_mol2(f, name='dendrobine')
            <generator object Structure.yield_from_mol2 at ...>
        """

        mol2io = StringIO(input) if isinstance(input, str) else input

        for block in read_mol2(mol2io):
            _name = name or block.header.name
            res = cls(None, n_atoms=block.header.n_atoms, name=_name)

            for i, a in enumerate(block.atoms):
                res.coords[i] = a.xyz
                res.atoms[i].set_mol2_type(a.mol2_type)
                res.atoms[i].label = a.label
                # This is to take care of tripos atom charge block
                if chrg := a.attrib.pop("charge", None):
                    res.atoms[i].formal_charge = int(chrg)

                res.atoms[i].attrib = a.attrib

            for i, b in enumerate(block.bonds):
                res.append_bond(
                    bond := Bond(
                        res.atoms[b.a1 - 1],
                        res.atoms[b.a2 - 1],
                        attrib=b.attrib,
                    )
                )
                bond.set_mol2_type(b.mol2_type)

            if DistanceUnit[source_units] != DistanceUnit.Angstrom:
                res.scale(DistanceUnit[source_units].value)

            if block.header.chrg_type != "NO_CHARGES" and hasattr(
                res, "atomic_charges"
            ):
                res.atomic_charges = [a.charge for a in block.atoms]

            yield res

    def dump_mol2(self, stream: StringIO) -> None:
        """Dumps the mol2 block into the output stream

        Parameters
        ----------
        _stream : StringIO, optional
            Output stream, by default None

        Examples
        -------
        The Molecule class inherits dump_mol2()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> with open('dendrobine.mol2', 'w') as f:
            >>>     dendrobine.dump_mol2(f)
            # Produced with molli package
            @<TRIPOS>MOLECULE
            dendrobine
            ...
        If desired, one can work directly with Structure class instead
            >>> dendrobine = ml.Structure.load_mol2(ml.files.dendrobine_mol2)
            >>> with open('dendrobine.mol2', 'w') as f:
            >>>     dendrobine.dump_mol2(f)
            # Produced with molli package
            @<TRIPOS>MOLECULE
            dendrobine
            ...
        """
        if hasattr(self, "name"):
            name = self.name
        else:
            name = "unknown"

        stream.write(f"# Produced with molli package\n")
        stream.write(
            f"@<TRIPOS>MOLECULE\n{name}\n{self.n_atoms} {self.n_bonds} 0 0"
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

    @classmethod
    def load_mol2(
        cls: type[Structure],
        input: str | Path | IO,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> Structure:
        """_summary_

        Parameters
        ----------
        cls : type[Structure]
            Class to be loaded into
        input : str | Path | IO
            File path, string, or stream
        name : str, optional
            Name for Structure, by default None
        source_units : str, optional
            Units to be used in loading, by default "Angstrom"

        Returns
        -------
        Structure
            Returns Structure

        Examples
        -------
        The Molecule class inherits load_mol2()
            >>> ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            Molecule(name='dendrobine', formula='C16 H25 N1 O2')
        If desired, one can work directly with Structure class instead
            >>> ml.Structure.load_mol2(ml.files.dendrobine_mol2)
            Structure(name='dendrobine', formula='C16 H25 N1 O2')
        """

        if isinstance(input, str | Path):
            stream = open(input, "rt")
        else:
            stream = input

        with stream:
            res = next(
                cls.yield_from_mol2(
                    stream,
                    name=name,
                    source_units=source_units,
                )
            )

        return res

    @classmethod
    def loads_mol2(
        cls: type[Structure],
        input: str,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> Structure:
        """Loads mol2 from a string

        Parameters
        ----------
        cls : type[Structure]
            Class to be loaded into
        input : str
            Mol2 block as string
        name : str, optional
            Name for Structure, by default None
        source_units : str, optional
            Units to be used in loading, by default "Angstrom"

        Returns
        -------
        Structure
            Returns Structure

        Examples
        -------
        The Molecule class inherits loads_mol2()
            >>> with open(ml.files.dendrobine_mol2, 'r') as f:
            >>>     ml.Molecule.loads_mol2(f.read())
            Molecule(name='dendrobine', formula='C16 H25 N1 O2')
        If desired, one can work directly with Structure class instead
            >>> with open(ml.files.dendrobine_mol2, 'r') as f:
            >>>     ml.Structure.loads_mol2(f.read())
            Structure(name='dendrobine', formula='C16 H25 N1 O2')
        """

        stream = StringIO(input)
        with stream:
            res = next(
                cls.yield_from_mol2(
                    stream,
                    name=name,
                    source_units=source_units,
                )
            )

        return res

    @classmethod
    def load_all_mol2(
        cls: type[Structure],
        input: str | Path | IO,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> List[Structure]:
        """This function loads all mol2 files from the input

        Parameters
        ----------
        cls : type[Structure]
            Class to be loaded into
        input : str | Path | IO
            File path, string, or stream
        name : str, optional
            Name for Structure, by default None
        source_units : str, optional
            Units to be used in loading, by default "Angstrom"

        Returns
        -------
        List[Structure]
            Returns list of Structures

        Examples
        -------
        The Molecule class inherits load_all_mol2()
            >>> ml.Molecule.load_all_mol2(ml.files.pentane_confs_mol2)
            [Molecule(name='pentane', formula='C5 H12'),...
        If desired, one can work directly with Structure class instead
            >>> ml.Structure.load_all_mol2(ml.files.pentane_confs_mol2)
            [Structure(name='pentane', formula='C5 H12'),...
        """
        """Load all components in a mol2 file from a multimol2 file"""
        if isinstance(input, str | Path):
            stream = open(input, "rt")
        else:
            stream = input

        with stream:
            res = list(
                cls.yield_from_mol2(
                    stream,
                    name=name,
                    source_units=source_units,
                )
            )

        return res

    @classmethod
    def loads_all_mol2(
        cls: type[Structure],
        input: str,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> List[Structure]:
        """This loads all mol2 files from the input string

        Parameters
        ----------
        cls : type[CartesianGeometry]
            Class to be loaded into
        input : str
            Mol2 Block as a string
        name : str, optional
            Name for Structure, by default None
        source_units : str, optional
            Units to be used in loading, by default "Angstrom"

        Returns
        -------
        List[Structure]
            Returns a list of Structures

        Examples
        -------
        The Molecule class inherits loads_all_mol2()
            >>> with open(ml.files.pentane_confs_mol2, 'r') as f:
            >>>     ml.Molecule.loads_all_mol2(f.read())
            [Molecule(name='pentane', formula='C5 H12'),...
        If desired, one can work directly with Structure class instead
            >>> with open(ml.files.pentane_confs_mol2, 'r') as f:
            >>>     ml.Structure.loads_all_mol2(f.read())
            [Structure(name='pentane', formula='C5 H12'),...
        """

        stream = StringIO(input)
        with stream:
            res = list(
                cls.yield_from_mol2(
                    stream,
                    name=name,
                    source_units=source_units,
                )
            )

        return res

    def dumps_mol2(self) -> str:
        """Dumps the mol2 block as a string

        Returns
        -------
        str
            The mol2 block

        Examples
        -------
        The Molecule class inherits dumps_mol2()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.dumps_mol2()
            # Produced with molli package
            @<TRIPOS>MOLECULE
            dendrobine
            ...
        If desired, one can work directly with Structure class instead
            >>> dendrobine = ml.Structure.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.dumps_mol2()
            # Produced with molli package
            @<TRIPOS>MOLECULE
            dendrobine
            ...
        """
        """
        This returns a mol2 file as a string
        """
        with StringIO() as stream:
            self.dumps_mol2(stream)
            return stream.getvalue()

    @classmethod
    def from_dict(self): ...

    @classmethod
    def join(
        cls,
        struct1: Structure,
        struct2: Structure,
        _a1: AtomLike,
        _a2: AtomLike,
        *,
        dist: float = None,
        optimize_rotation: bool | int = False,
        name: str = None,
        charge: float = None,
        mult: float = None,
        btype: BondType = BondType.Single,
        bstereo: BondStereo = BondStereo.Unknown,
        bforder: float = 1.0,
    ) -> Structure:
        """This can be used to join two structures together at individual atoms

        Parameters
        ----------
        struct1 : Structure
            First structure
        struct2 : Structure
            Second structure
        _a1 : AtomLike
            First atom
        _a2 : AtomLike
            Second atom
        dist : float, optional
            Distance for two structures to be joined at, by default None
        optimize_rotation : bool | int, optional
            Rotates structures if there is expected to be overlapping
            van der Waals radii, by default False
        name : str, optional
            Name of the new structure, by default None
        charge : float, optional
            Charge of the new structure, by default None
        mult : float, optional
            Multiplicity of the new structure, by default None
        btype : BondType, optional
            Type of bond formed, by default BondType.Single
        bstereo : BondStereo, optional
            Geometry of bond formed, by default BondStereo.Unknown
        bforder : float, optional
            Fractional order of bond formed, by default 1.0

        Returns
        -------
        Structure
            Returns a structure joined at the the atoms of interest

        Examples
        -------
        The Molecule class inherits join()
            >>> mol1 = ml.Molecule.load_mol2('mol1_w_attachment_point.mol2')
            >>> ap1, = mol1.get_attachment_points()
            >>> mol2 = ml.Molecule.load_mol2('mol1_w_attachment_point.mol2')
            >>> ap2, = mol2.get_attachment_points()
            >>> res = ml.Molecule.join(mol1, mol2, ap1, ap2, optimize_rotation=True)
        If desired, one can work directly with Structure class instead
            >>> mol1 = ml.Structure.load_mol2('mol1_w_attachment_point.mol2')
            >>> ap1, = mol1.get_attachment_points()
            >>> mol2 = ml.Structure.load_mol2('mol1_w_attachment_point.mol2')
            >>> ap2, = mol2.get_attachment_points()
            >>> res = ml.Structure.join(mol1, mol2, ap1, ap2, optimize_rotation=True)
        """
        assert struct1.n_bonds_with_atom(_a1) == 1, (
            f"{struct1.get_atom(_a1)} does not seem to be a valid attachment point."
            f" {struct1.n_bonds_with_atom(_a1)=}"
        )
        assert struct2.n_bonds_with_atom(_a2) == 1, (
            f"{struct1.get_atom(_a2)} does not seem to be a valid attachment point."
            f" {struct1.n_bonds_with_atom(_a2)=}"
        )

        # Atoms that are bonded to the attachment points
        a1 = struct1.get_atom(_a1)
        a2 = struct2.get_atom(_a2)
        a1r = next(struct1.connected_atoms(_a1))
        a2r = next(struct2.connected_atoms(_a2))

        atoms = [a for a in chain(struct1.atoms, struct2.atoms) if a not in {a1, a2}]
        charge = charge or struct1.charge + struct2.charge
        mult = mult or struct1.mult + struct2.mult - 1

        result = cls(
            atoms,
            name=name,
            copy_atoms=True,
            charge=charge,
            mult=mult,
        )

        atom_map = dict(zip(atoms, result.atoms))
        for j, b in enumerate(chain(struct1.bonds, struct2.bonds)):
            if a1 not in b and a2 not in b:
                result.append_bond(
                    b.evolve(a1=atom_map[b.a1], a2=atom_map[b.a2], parent=result)
                )

        result.append_bond(
            nb := Bond(
                result.atoms[atoms.index(a1r)],
                result.atoms[atoms.index(a2r)],
                btype=btype,
                stereo=bstereo,
                f_order=bforder,
            )
        )

        loc1 = ~np.array([a == a1 for a in struct1.atoms])
        loc2 = ~np.array([a == a2 for a in struct2.atoms])

        v1 = struct1.vector(a1r, a1)
        v2 = struct2.vector(a2r, a2)

        r1 = struct1.get_atom_coord(a1r)
        r2 = struct2.get_atom_coord(a2r)

        rotation = rotation_matrix_from_vectors(v2, -v1, tol=1e-6)
        translation = v1 * (dist or nb.expected_length or 1.5) / np.linalg.norm(v1)

        c1 = struct1.coords[loc1] - r1
        c2 = (struct2.coords[loc2] - r2) @ rotation + translation

        if optimize_rotation:
            c2 = c2 @ _optimize_rotation(c1, c2, v1, resolution=12)

        # c2 += translation

        result.coords = np.vstack((c1, c2))
        return result

    @classmethod
    def concatenate(cls, *structs: Structure) -> Structure:
        """Concatenates atom and bond tables of structures

        Returns
        -------
        Structure
            Returns concatenated structure

        Examples
        -------
        The Molecule class inherits concatenate()
            >>> mol1 = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> mol2 = ml.Molecule.load_mol2(ml.files.benzene_mol2)
            >>> ml.Molecule.concatenate(mol1, mol2)
            Molecule(name='unknown', formula='C22 H31 N1 O2')
        If desired, one can work directly with Structure class instead
            >>> mol1 = ml.Structure.load_mol2(ml.files.dendrobine_mol2)
            >>> mol2 = ml.Structure.load_mol2(ml.files.benzene_mol2)
            >>> ml.Structure.concatenate(mol1, mol2)
            Structure(name='unknown', formula='C22 H31 N1 O2')
        """
        source_atoms = list(chain.from_iterable(x.atoms for x in structs))
        res = cls(source_atoms, copy_atoms=True)

        atom_map = {source_atoms[i]: res.atoms[i] for i in range(res.n_atoms)}

        for j, b in enumerate(chain.from_iterable(x.bonds for x in structs)):
            res.append_bond(b.evolve(a1=atom_map[b.a1], a2=atom_map[b.a2]))

        res.coords = np.vstack([x.coords for x in structs])

        res.charge = np.sum(np.fromiter((s.charge for s in structs), dtype="int32"))
        res.mult = np.sum(np.fromiter((s.mult for s in structs), dtype="int32")) - 1

        return res

    def extend(self, other: Structure) -> None:
        """Currently Not Implemented

        This extends current structure with the copied atoms, bonds
        and coordinates from another

        Parameters
        ----------
        other : Structure
            Structure to extend with

        Raises
        ------
        NotImplementedError
            _description_
        """
        """This extends current structure with the copied atoms, bonds and coordinates from another"""
        raise NotImplementedError("Extending Structures is Currently Not Implemented")

    def substructure(self, atoms: Iterable[AtomLike]) -> Substructure:
        """Creates a substructure from a subset of atoms

        Parameters
        ----------
        atoms : Iterable[AtomLike]
            Subset of atoms to create Substructure from

        Returns
        -------
        Substructure
            Returns a Substructure from a parent Structure

        Examples
        -------
        The Molecule class inherits substructure()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.substructure([0,1,2])
            Substructure(parent=Molecule(name='dendrobine', ...), atoms=[0,1,2])
        If desired, one can work directly with Structure class instead
            >>> dendrobine = ml.Structure.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.substructure([0,1,2])
            Substructure(parent=Structure(name='dendrobine', ...), atoms=[0,1,2])
        """
        return Substructure(self, list(atoms))

    @property
    def heavy(self) -> Substructure:
        """Returns a substructure containing only heavy atoms.

        Returns
        -------
        Substructure
            The substructure containing only heavy atoms.

        Examples
        -------
        The Molecule class inherits heavy
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.heavy
            Substructure(parent=Molecule(name='dendrobine', ...), atoms=[0,1,...])
        If desired, one can work directly with Structure class instead
            >>> dendrobine = ml.Structure.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.heavy
            Substructure(parent=Structure(name='dendrobine', ...), atoms=[0,1,...])
        """

        return Substructure(self, [a for a in self.atoms if a.element != Element.H])

    def bond_length(self, b: Bond) -> float:
        """Returns the length of a bond.

        Parameters
        ----------
        b : Bond
            The bond to measure.

        Returns
        -------
        float
            The length of the bond.

        Examples
        -------
        The Molecule class inherits bond_length()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> bond = dendrobine.get_bond(0)
            >>> dendrobine.bond_length(bond)
            1.0956031261364676
        If desired, one can work directly with Structure class instead
            >>> dendrobine = ml.Structure.load_mol2(ml.files.dendrobine_mol2)
            >>> bond = dendrobine.get_bond(0)
            >>> dendrobine.bond_length(bond)
            1.0956031261364676
        """

        return self.distance(b.a1, b.a2)

    def bond_vector(self, b: Bond) -> np.ndarray:
        """Returns the vector between the two atoms in a bond.

        Parameters
        ----------
        b : Bond
             The bond to measure.

        Returns
        -------
        np.ndarray
            The vector between the two atoms in the bond.

        Examples
        -------
        The Molecule class inherits bond_vector()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> bond = dendrobine.get_bond(0)
            >>> dendrobine.bond_vector(bond)
            array([-0.6747, -0.6486,  0.5696])
        If desired, one can work directly with Structure class instead
            >>> dendrobine = ml.Structure.load_mol2(ml.files.dendrobine_mol2)
            >>> bond = dendrobine.get_bond(0)
            >>> dendrobine.bond_vector(bond)
            array([-0.6747, -0.6486,  0.5696])
        """

        i1, i2 = map(self.get_atom_index, (b.a1, b.a2))
        return self.coords[i2] - self.coords[i1]

    def bond_coords(self, b: Bond) -> tuple[np.ndarray]:
        """Returns the coordinates of the two atoms in a bond.

        Parameters
        ----------
        b : Bond
            The bond to measure.

        Returns
        -------
        tuple[np.ndarray]
            The coordinates of the two atoms in the bond.

        Examples
        -------
        The Molecule class inherits bond_coords()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> bond = dendrobine.get_bond(0)
            >>> dendrobine.bond_coords(bond)
            array([[ 1.0232, -0.452 , -5.421 ], [ 0.3485, -1.1006, -4.8514]])
        If desired, one can work directly with Structure class instead
            >>> dendrobine = ml.Structure.load_mol2(ml.files.dendrobine_mol2)
            >>> bond = dendrobine.get_bond(0)
            >>> dendrobine.bond_coords(bond)
            array([[ 1.0232, -0.452 , -5.421 ], [ 0.3485, -1.1006, -4.8514]])
        """

        return self.coord_subset((b.a1, b.a2))

    def __or__(self, other: Structure) -> Structure:
        """This function concatenates two structures

        Parameters
        ----------
        other : Structure
            The other structure to concatenate with.

        Returns
        -------
        Structure
            The concatenated structure.
        Examples
        -------
        The Molecule class inherits concatenate()
            >>> mol1 = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> mol2 = ml.Molecule.load_mol2(ml.files.benzene_mol2)
            >>> mol1 | mol2
            Structure(name='unknown', formula='C22 H31 N1 O2')
        If desired, one can work directly with Structure class instead
            >>> mol1 = ml.Structure.load_mol2(ml.files.dendrobine_mol2)
            >>> mol2 = ml.Structure.load_mol2(ml.files.benzene_mol2)
            >>> mol1 | mol2
            Structure(name='unknown', formula='C22 H31 N1 O2')
        """

        return Structure.concatenate(self, other)

    def perceive_atom_properties(self, _a: AtomLike) -> None:
        """Currently Not Implemented

        This function analyzes atomic properties

        Parameters
        ----------
        _a : AtomLike
            This is the atom for analysis
        """

        raise NotImplementedError(
            "perceive_atom_properties is Currently Not Implemented"
        )
        a = self.get_atom(_a)
        n_bonds = self.n_bonds_with_atom(a)
        max_bond_order = max(b.order for b in self.bonds_with_atom(_a))

        match max_bond_order, n_bonds:
            case 1 if a.element.group in range(14, 19):
                a.atype = AtomType.sp3

            case 14, 2:
                a.atype = AtomType.sp2

    def perceive_bond_properties(self) -> None:
        """Currently Not Implemented

        This function analyzes bond properties

        """
        raise NotImplementedError(
            "perceive_bond_properties is Currently Not Implemented"
        )

    def rotate_dihedral(self, atoms: tuple[AtomLike], target_angle: float):
        """This procedure rotates the substructure"""
        dihedral = self.dihedral(*atoms)
        rotation_angle = target_angle - dihedral
        ax = self.vector(atoms[1], atoms[2])
        origin = self.get_atom_coord(atoms[1])

        # Define the rotation matrix
        R = rotation_matrix_from_axis(ax, rotation_angle)

        # substructure to be rotated
        substruct = self.substructure(self.yield_bfs(atoms[1], atoms[2]))

        substruct.translate(-origin)
        substruct.transform(R)
        substruct.translate(origin)

    def remove_substituent(self, a1: AtomLike, a2: AtomLike, *, ap_label: str = None):
        """This removes the substituent from the"""
        c2 = self.get_atom_coord(a2)

        for a in tuple(self.yield_bfs(a1, a2)):
            self.del_atom(a)

        self.add_atom(
            a := Atom(
                element=Element.Unknown,
                atype=AtomType.AttachmentPoint,
                label=ap_label,
            ),
            c2,
        )
        self.connect(a1, a)

    def split(
        self,
        a1: AtomLike,
        a2: AtomLike,
        *,
        ap1_label: str = None,
        ap2_label: str = None,
    ):
        """This is a stub of the function that splits"""
        raise NotImplementedError

    def del_atom(self, _a: AtomLike):
        """Deletes an atom from the Structure

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
        If desired, one can work directly with Structure class instead
            >>> Structure = ml.Structure(dendrobine)
            >>> Structure.del_atom(0)
            >>> Structure.get_atom(0)
            Atom(element=C, isotope=None, label='C', formal_charge=0, formal_spin=0)
        """
        a = self.get_atom(_a)
        super().del_atom(a)

    def add_implicit_hydrogens(self, *atoms: AtomLike) -> None:
        """This function adds implicit hydrogens to all specified atoms.
        By default, it will add implicit hydrogens to all atoms if
        necessary.

        Examples
        -------
        The Molecule class inherits add_implicit_hydrogens()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.n_atoms
            44
            >>> dendrobine.add_implicit_hydrogens()
            >>> dendrobine.n_atoms
            44
        If desired, one can work directly with Structure class instead
            >>> dendrobine = ml.Structure.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.n_atoms
            44
            >>> dendrobine.add_implicit_hydrogens()
            >>> dendrobine.n_atoms
            44
        """

        from molli.math.polyhedra import TETRAHEDRON
        from molli.math import rotation_matrix_from_vectors, mean_plane

        if len(atoms) == 0:
            atoms = [a for a in self.atoms if a.element.group in range(13, 17)]

        for a in atoms:
            if (hs_to_add := a.attrib.pop("__implicit_hydrogens", None)) is not None:
                # This only happens if there was a pre-defined attribute containing the number of implicit hydrogens.
                # Usually happens because of CDXML parsing algorithm
                pass
            else:
                electrons = a.valence_electrons - a.formal_charge - abs(a.formal_spin)
                bonded = ceil(self.bonded_valence(a))
                hs_to_add = max(4 - abs(4 - electrons) - bonded, 0)

            if hs_to_add > 0:

                neighbors = list(
                    a
                    for a in self.connected_atoms(a)
                    if a.atype != AtomType.CoordinationCenter
                )
                a_coord = self.get_atom_coord(a)
                if len(neighbors) == 3:
                    vec = mean_plane(self.coord_subset(neighbors))
                    cent = np.average(self.coord_subset(neighbors), axis=0)
                    align = np.dot(vec, cent - a_coord)
                    if (
                        abs(align) > 0.05
                    ):  # Note that this threshold is completely arbitrary.
                        vec *= align
                else:
                    vec = np.average(self.coord_subset(neighbors) - a_coord, axis=0)
                vec /= np.linalg.norm(vec)
                L = a.cov_radius_1 + Element.H.cov_radius_1

                if hs_to_add == 1:
                    newh = Atom("H")
                    newbond = Bond(a, newh)
                    self.add_atom(newh, a_coord - vec * L)
                    self.append_bond(newbond)

                elif hs_to_add == 2:
                    if len(neighbors) == 2:
                        r1, r2 = self.coord_subset(neighbors) - a_coord
                        z = np.cross(
                            r1, r2
                        )  # this is a vector that is orthogonal to neighbors
                        z /= np.linalg.norm(z)
                    else:
                        z = np.cross(vec, [0.0, 0.0, 1.0])
                        z /= np.linalg.norm(z)

                    c1 = a_coord - (vec * 0.5736 + z * 0.8192) * L
                    c2 = a_coord - (vec * 0.5736 - z * 0.8192) * L
                    self.add_atom(h1 := Atom("H"), c1)
                    self.add_atom(h2 := Atom("H"), c2)
                    self.append_bond(Bond(a, h1))
                    self.append_bond(Bond(a, h2))

                elif hs_to_add == 3:
                    R = rotation_matrix_from_vectors(TETRAHEDRON[0], vec)
                    tet_rot = TETRAHEDRON @ R * L + a_coord

                    for c in tet_rot[1:]:
                        self.add_atom(h := Atom("H"), c)
                        self.append_bond(Bond(a, h))


class Substructure(Structure):
    """This class represents a substructure of a structure. It pulls the
    atoms and bonds from the parent structure, and allows for manipulation of
    the a subset of atoms within the initial structure.
    """

    def __init__(self, parent: Structure, atoms: Iterable[AtomLike]):
        self._parent = parent
        self._atoms = [parent.get_atom(a) for a in atoms]
        self._bonds = []
        self.attrib = parent.attrib.copy()

        for b in parent.bonds:
            if b.a1 in self.atoms and b.a2 in self.atoms:
                self._bonds.append(b)

    def yield_parent_atom_indices(
        self, atoms: Iterable[AtomLike]
    ) -> Generator[int, None, None]:
        """This function yields the indices of the atoms in the parent structure.

        Parameters
        ----------
        atoms : Iterable[AtomLike]
            The atoms to yield the indices of.

        Yields
        ------
        Generator[int, None, None]
            The indices of the atoms in the parent structure.

        Examples
        -------
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.heavy
            Substructure(parent=Molecule(name='dendrobine', ...), atoms=[0,1,...])
            >>> substruc.yield_parent_atom_indices(struc.yield_atoms_by_element("H"))
            <generator object Substructure.yield_parent_atom_indices at ...>
        """

        yield from map(self._parent.get_atom_index, atoms)

    def __repr__(self):
        return f"""{type(self).__name__}(parent={self._parent!r}, atoms={self.parent_atom_indices!r})"""

    @property
    def parent_atom_indices(self) -> list[int]:
        """Returns the indices of the atoms in the parent structure.

        Returns
        -------
        list[int]
            The indices of the atoms in the parent structure.

        Examples
        -------
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.heavy
            Substructure(parent=Molecule(name='dendrobine', ...), atoms=[0,1,...])
            >>> substruc.parent_atom_indices
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22]
        """

        return list(self.yield_parent_atom_indices(self._atoms))

    @property
    def coords(self) -> np.ndarray:
        """Returns the coordinates of the atoms in the substructure.

        Returns
        -------
        np.ndarray
            The coordinates of the atoms in the substructure.

        Examples
        -------
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.coords
            (44,3)
            >>> dendrobine.heavy.coords
            array([[ 1.2960e+00, -2.3190e-01,  1.2670e+00],...
            >>> dendrobine.heavy.coords.shape
            (19,3)
        """

        return self._parent.coords[self.parent_atom_indices]

    @coords.setter
    def coords(self, other):
        self._parent.coords[self.parent_atom_indices] = other

    def __or__(self, other: Substructure | Structure) -> Structure | Substructure:
        """This function concatenates two structures or substructures

        Parameters
        ----------
        other : Substructure | Structure
            The other structure or substructure to concatenate with.

        Returns
        -------
        Structure | Substructure
            The concatenated structure or substructure

        Examples
        -------
        The Molecule class inherits concatenate()
            >>> mol1 = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> mol2 = ml.Molecule.load_mol2(ml.files.benzene_mol2)
            >>> mol1.heavy | mol2.heavy
            Structure(name='unknown', formula='C22 N1 O2')
        """

        if isinstance(other, Substructure) and other.parent == self._parent:
            return Substructure(self._parent, chain(self.atoms, other.atoms))
        else:
            return Structure.concatenate(self, other)
