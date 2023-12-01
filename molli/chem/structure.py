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
from ..math import rotation_matrix_from_vectors, _optimize_rotation
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
    """Structure is a simple amalgamation of the concepts of CartesianGeometry and Connectivity"""

    # Improve efficiency of attribute access
    __slots__ = ("_name", "_atoms", "_bonds", "_coords", "charge", "mult")

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
        """Structure."""
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
    ):
        mol2io = StringIO(input) if isinstance(input, str) else input

        for block in read_mol2(mol2io):
            _name = name or block.header.name
            res = cls(None, n_atoms=block.header.n_atoms, name=_name)

            for i, a in enumerate(block.atoms):
                res.coords[i] = a.xyz
                res.atoms[i].set_mol2_type(a.mol2_type)
                res.atoms[i].label = a.label

            for i, b in enumerate(block.bonds):
                res.append_bond(
                    bond := Bond(
                        res.atoms[b.a1 - 1],
                        res.atoms[b.a2 - 1],
                    )
                )
                bond.set_mol2_type(b.mol2_type)

            if DistanceUnit[source_units] != DistanceUnit.Angstrom:
                res.scale(DistanceUnit[source_units].value)

            if block.header.chrg_type != "NO_CHARGES" and hasattr(res, "atomic_charges"):
                res.atomic_charges = [a.charge for a in block.atoms]

            yield res

    def dump_mol2(self, _stream: StringIO = None):
        if _stream is None:
            stream = StringIO()

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

        if _stream is None:
            return stream.getvalue()

    @classmethod
    def load_mol2(
        cls: type[CartesianGeometry],
        input: str | Path | IO,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> CartesianGeometry:
        """Load mol2 from a file stream or file"""
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
        cls: type[CartesianGeometry],
        input: str,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> CartesianGeometry:
        """Load mol2 file from string"""
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
        cls: type[CartesianGeometry],
        input: str | Path | IO,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> List[CartesianGeometry]:
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
        cls: type[CartesianGeometry],
        input: str | Path | IO,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> List[CartesianGeometry]:
        """LOAD ALL MOL2"""
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
        """
        This returns a mol2 file as a string
        """
        return self.dump_mol2()

    @classmethod
    def from_dict(self):
        ...

    @classmethod
    def join(
        cls: type[Structure],
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
    ):
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
                result.append_bond(b.evolve(a1=atom_map[b.a1], a2=atom_map[b.a2]))

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
    def concatenate(cls: type[Structure], *structs: Structure) -> Structure:
        source_atoms = list(chain.from_iterable(x.atoms for x in structs))
        res = cls(source_atoms, copy_atoms=True)

        atom_map = {source_atoms[i]: res.atoms[i] for i in range(res.n_atoms)}

        for j, b in enumerate(chain.from_iterable(x.bonds for x in structs)):
            res.append_bond(b.evolve(a1=atom_map[b.a1], a2=atom_map[b.a2]))

        res.coords = np.vstack([x.coords for x in structs])

        res.charge = np.sum(s.charge for s in structs)
        res.mult = np.sum(s.mult for s in structs) - 1

        return res

    def extend(self, other: Structure) -> None:
        """This extends current structure with the copied atoms, bonds and coordinates from another"""
        raise NotImplementedError

    def substructure(self, atoms: Iterable[AtomLike]) -> Substructure:
        return Substructure(self, list(atoms))

    @property
    def heavy(self) -> Substructure:
        return Substructure(self, [a for a in self.atoms if a.element != Element.H])

    def bond_length(self, b: Bond) -> float:
        return self.distance(b.a1, b.a2)

    def bond_vector(self, b: Bond) -> np.ndarray:
        i1, i2 = map(self.get_atom_index, (b.a1, b.a2))
        return self.coords[i2] - self.coords[i1]

    def bond_coords(self, b: Bond) -> tuple[np.ndarray]:
        return self.coord_subset((b.a1, b.a2))

    def __or__(self, other: Structure) -> Structure:
        return Structure.concatenate(self, other)

    def perceive_atom_properties(self, _a: AtomLike) -> None:
        """# `perceive_atom_properties`
        This function analyzes atomic types

        ## Returns

        `_type_`
            _description_

        ## Yields

        `_type_`
            _description_
        """
        a = self.get_atom(_a)
        n_bonds = self.n_bonds_with_atom(a)
        max_bond_order = max(b.order for b in self.bonds_with_atom(_a))

        match max_bond_order, n_bonds:
            case 1 if a.element.group in range(14, 19):
                a.atype = AtomType.MainGroup_sp3

            case 14, 2:
                a.atype = AtomType.MainGroup_sp2

    def perceive_bond_properties(self) -> None:
        """# `perceive_bond_properties`
        This function analyzes bond properties

        ## Returns

        `_type_`
            _description_

        ## Yields

        `_type_`
            _description_
        """
        raise NotImplementedError

    def del_atom(self, _a: AtomLike):
        a = self.get_atom(_a)
        super().del_atom(a)

    def add_implicit_hydrogens(self, *atoms: AtomLike):
        """
        This function adds implicit hydrogens to all specified atoms.
        TODO: rewrite with optimization (!)
        """
        from molli.math.polyhedra import TETRAHEDRON
        from molli.math import rotation_matrix_from_vectors, mean_plane

        if len(atoms) == 0:
            atoms = [a for a in self.atoms if a.element.group in range(13, 17)]

        for a in atoms:
            if (diff := a.implicit_valence - int(self.bonded_valence(a))) >= 0:
                neighbors = list(self.connected_atoms(a))
                a_coord = self.get_atom_coord(a)
                if len(neighbors) == 3:
                    vec = mean_plane(self.coord_subset(neighbors))
                    cent = np.average(self.coord_subset(neighbors), axis=0)
                    align = np.dot(vec, cent - a_coord)
                    if abs(align) > 0.05:  # Note that this threshold is completely arbitrary.
                        vec *= align
                else:
                    vec = np.average(self.coord_subset(neighbors) - a_coord, axis=0)
                vec /= np.linalg.norm(vec)
                L = a.cov_radius_1 + Element.H.cov_radius_1

                if diff == 1:
                    newh = Atom("H")
                    newbond = Bond(a, newh)
                    self.add_atom(newh, a_coord - vec * L)
                    self.append_bond(newbond)

                elif diff == 2:
                    if len(neighbors) == 2:
                        r1, r2 = self.coord_subset(neighbors) - a_coord
                        z = np.cross(r1, r2)  # this is a vector that is orthogonal to neighbors
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

                elif diff == 3:
                    R = rotation_matrix_from_vectors(TETRAHEDRON[0], vec)
                    tet_rot = TETRAHEDRON @ R * L + a_coord

                    for c in tet_rot[1:]:
                        self.add_atom(h := Atom("H"), c)
                        self.append_bond(Bond(a, h))


class Substructure(Structure):
    def __init__(self, parent: Structure, atoms: Iterable[AtomLike]):
        self._parent = parent
        self._atoms = [parent.get_atom(a) for a in atoms]
        self._bonds = []

        for b in parent.bonds:
            if b.a1 in self.atoms and b.a2 in self.atoms:
                self._bonds.append(b)

    def yield_parent_atom_indices(self, atoms: Iterable[AtomLike]) -> Generator[int, None, None]:
        yield from map(self._parent.get_atom_index, atoms)

    def __repr__(self):
        return f"""{type(self).__name__}(parent={self._parent!r}, atoms={self.parent_atom_indices!r})"""

    @property
    def parent_atom_indices(self):
        return list(self.yield_parent_atom_indices(self._atoms))

    @property
    def coords(self):
        return self._parent.coords[self.parent_atom_indices]

    @coords.setter
    def coords(self, other):
        self._parent.coords[self.parent_atom_indices] = other

    def __or__(self, other: Substructure | Structure):
        if isinstance(other, Substructure) and other.parent == self._parent:
            return Substructure(self._parent, chain(self.atoms, other.atoms))
        else:
            return Structure.concatenate(self, other)
