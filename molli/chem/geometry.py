from __future__ import annotations

from . import (
    Element,
    ElementLike,
    Atom,
    AtomLike,
    AtomType,
    Bond,
    Promolecule,
    PromoleculeLike,
    Connectivity,
)
from ..parsing import read_xyz
from typing import (
    Any,
    List,
    Iterable,
    Generator,
    TypeVar,
    Generic,
    Callable,
    IO,
)
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from io import StringIO, BytesIO, IOBase
from functools import wraps
from pathlib import Path
import math


class DistanceUnit(Enum):
    """
    This small enumeration lists commonly used distance units
    """

    A = 1.0
    Angstrom = A
    Bohr = 1.88973
    au = Bohr
    fm = 100_000.0
    pm = 100.0
    nm = 0.100


def _angle(v1, v2):
    """computes the angle formed by two vectors"""
    dt = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(dt)


class CartesianGeometry(Promolecule):
    """
    Cartesian Geometry Class
    Stores molecular geometry in ANGSTROM floating points.
    This version is generalizable to arbitrary coordinates and data types
    """

    _coords_dtype = np.float64

    def __init_subclass__(cls, coords_dtype=np.float64, **kwds) -> None:
        super().__init_subclass__(**kwds)
        cls._coords_dtype = np.dtype(coords_dtype)

    def __init__(
        self,
        other: Promolecule = None,
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
        # Type of coordinates
        super().__init__(
            other,
            n_atoms=n_atoms,
            name=name,
            copy_atoms=copy_atoms,
            charge=charge,
            mult=mult,
            **kwds,
        )
        self._coords = np.empty((self.n_atoms, 3), self._coords_dtype)

        if isinstance(other, CartesianGeometry):
            self.coords = other.coords
        else:
            self.coords = coords if coords is not None else np.nan

    # ADD METHODS TO OVERRIDE ADDING ATOMS!

    def add_atom(self, a: Atom, coord: ArrayLike):
        super().append_atom(a)
        _coord = np.array(coord, dtype=self._coords_dtype)
        if not _coord.shape == (3,):
            raise ValueError("Inappropriate coordinates for atom (interpreted as {_coord})")
        self._coords = np.append(self._coords, [coord], axis=0)

    def new_atom(
        self,
        element: Element = ...,
        isotope: int = None,
        coord: ArrayLike = [0, 0, 0],
        **kwargs,
    ) -> Atom:
        _a = Atom(element, isotope, **kwargs)
        self.add_atom(_a, coord)
        return _a

    @property
    def coords(self):
        """Set of atomic positions in shape (n_atoms, 3)"""
        return self._coords

    @coords.setter
    def coords(self, other: ArrayLike):
        self._coords[:] = other

    @property
    def coords_as_list(self):
        return self._coords.flatten().tolist()

    def extend(self, other: CartesianGeometry):
        """This extends current geometry with another one"""
        raise NotImplementedError

    def dump_xyz(self, output: StringIO, write_header: bool = True) -> None:
        """
        This dumps an xyz file into the output stream.
        """
        # Header need not be written in certain files
        # Like ORCA inputs
        if write_header:
            if hasattr(self, "name"):
                comment = f"{self.name} [produced with molli]"
            else:
                comment = f"{self!r} [produced with molli]"
            output.write(f"{self.n_atoms}\n{comment}\n")

        for i in range(self.n_atoms):
            s = self.atoms[i].element.symbol
            x, y, z = self.coords[i]
            output.write(f"{s:<5} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    def dumps_xyz(self, write_header: bool = True) -> str:
        """
        This returns an xyz file as a string
        """
        # Header need not be written in certain files
        # Like ORCA inputs
        res = StringIO()
        self.dump_xyz(res, write_header=write_header)

        return res.getvalue()

    @classmethod
    def load_xyz(
        cls: type[CartesianGeometry],
        input: str | Path | IO,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> CartesianGeometry:
        """# `load_xyz`
        This function loads a *single* xyz file into the current instance
        The input should be a stream or file name/path

        ## Parameters

        `cls: type[CartesianGeometry]`
            class instance

        `input: str | Path | IO`
            Input must be an IOBase instance (typically, an open file)
            Alternatively, a string or path will be considered as a path to file that will need to be opened.
        `source_units: str`, optional, default: `"Angstrom"`
            Source units should be one of: A == Angstrom, Bohr == au, fm, pm, nm
        """

        if isinstance(input, str | Path):
            stream = open(input, "rt")
        else:
            stream = input

        with stream:
            res = next(cls.yield_from_xyz(stream, name=name, source_units=source_units))

        return res

    @classmethod
    def loads_xyz(
        cls: type[CartesianGeometry],
        input: str,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> CartesianGeometry:
        """# `load_xyz`
        This function loads a *single* xyz file into the current instance
        The input should be a string instance

        ## Parameters

        `cls: type[CartesianGeometry]`
            class instance

        `input: str`
            Input must be an IOBase instance (typically, an open file)
            Alternatively, a string or path will be considered as a path to file that will need to be opened.
        `source_units: str`, optional, default: `"Angstrom"`
            Source units should be one of: A == Angstrom, Bohr == au, fm, pm, nm
        """
        stream = StringIO(input)
        with stream:
            res = next(cls.yield_from_xyz(stream, name=name, source_units=source_units))

        return res

    @classmethod
    def load_all_xyz(
        cls: type[CartesianGeometry],
        input: str | Path | IO,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> List[CartesianGeometry]:
        if isinstance(input, str | Path):
            stream = open(input, "rt")
        else:
            stream = input

        with stream:
            res = list(cls.yield_from_xyz(stream, name=name, source_units=source_units))

        return res

    @classmethod
    def loads_all_xyz(
        cls: type[CartesianGeometry],
        input: str | Path | IO,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> List[CartesianGeometry]:
        stream = StringIO(input)
        with stream:
            res = list(cls.yield_from_xyz(stream, name=name, source_units=source_units))

        return res

    @classmethod
    def yield_from_xyz(
        cls: type[CartesianGeometry],
        stream: StringIO,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ):
        for xyzblock in read_xyz(stream):
            geom = cls(n_atoms=xyzblock.n_atoms, coords=xyzblock.coords)
            for i, a in enumerate(xyzblock.atoms):
                if a.symbol == "*":
                    geom.atoms[i].atype = AtomType.Dummy
                    geom.atoms[i].element = Element.Unknown
                else:
                    geom.atoms[i].element = Element.get(a.symbol)

            if DistanceUnit[source_units] != DistanceUnit.Angstrom:
                geom.scale(DistanceUnit[source_units].value)

            geom.name = name or "unnamed"
            yield geom

    def scale(self, factor: float, allow_inversion=False):
        """
        Simple multiplication of all coordinates by a factor.
        Useful for unit conversion.
        """
        if factor < 0 and not allow_inversion:
            raise ValueError(
                "Scaling with a negative factor can only be performed with"
                " explicit `scale(factor, allow_inversion = True)`"
            )

        if factor == 0:
            raise ValueError("Scaling with a factor == 0 is not allowed.")

        self.coords *= factor

    def invert(self):
        """
        Coordinates are inverted wrt the origin. This also inverts the absolute stereochemistry
        """
        self.scale(-1, allow_inversion=True)

    def distance(self, a1: AtomLike, a2: AtomLike) -> float:
        i1, i2 = map(self.get_atom_index, (a1, a2))
        return np.linalg.norm(self.coords[i1] - self.coords[i2])

    def get_atom_coord(self, _a: AtomLike):
        return self.coords[self.get_atom_index(_a)]

    def vector(self, a1: AtomLike, a2: AtomLike | np.ndarray) -> np.ndarray:
        """Get a vector [a1 --> a2], where a2 can be a numpy array or another atom"""
        v1 = self.get_atom_coord(a1)
        v2 = self.get_atom_coord(a2) if isinstance(a2, AtomLike) else np.array(a2)

        return v2 - v1

    def distance_to_point(self, a: AtomLike, p: ArrayLike) -> float:
        return np.linalg.norm(self.vector(a, p))

    def angle(self, a1: AtomLike, a2: AtomLike, a3: AtomLike) -> float:
        """
        Compute an angle
        """
        i1, i2, i3 = map(self.get_atom_index, (a1, a2, a3))

        v1 = self.coords[i1] - self.coords[i2]
        v2 = self.coords[i3] - self.coords[i2]

        # dt = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # return np.arccos(dt)

        return _angle(v1, v2)

    def coord_subset(self, atoms: Iterable[AtomLike]) -> np.ndarray:
        indices = list(map(self.get_atom_index, atoms))
        return self.coords[indices]

    def dihedral(
        self,
        a1: AtomLike,
        a2: AtomLike,
        a3: AtomLike,
        a4: AtomLike,
    ) -> float:
        i1, i2, i3, i4 = map(self.get_atom_index, (a1, a2, a3, a4))

        u1 = self.coords[i2] - self.coords[i1]
        u2 = self.coords[i3] - self.coords[i2]
        u3 = self.coords[i4] - self.coords[i3]

        u2xu3 = np.cross(u2, u3)
        u1xu2 = np.cross(u1, u2)
        arg1 = np.linalg.norm(u2) * np.dot(u1, u2xu3)
        arg2 = np.dot(u1xu2, u2xu3)

        return np.arctan2(arg1, arg2)

    def translate(self, vector: ArrayLike):
        v = np.array(vector)
        self.coords += v[np.newaxis, :]

    def centroid(self) -> np.ndarray:
        """Return the centroid of the molecule"""
        return np.average(self.coords, axis=0)

    def rmsd(self, other: CartesianGeometry, validate_elements=True):
        if other.n_atoms != self.n_atoms:
            raise ValueError("Cannot compare geometries with different number of atoms")

        if validate_elements == True and self.elements != other.elements:
            raise ValueError("Cannot compare two molecules with different lists of elements")

        raise NotImplementedError

    def transform(self, _t_matrix: ArrayLike, /, validate=False):
        t_matrix = np.array(_t_matrix)
        self.coords = self.coords @ t_matrix

    def del_atom(self, _a: AtomLike):
        ai = self.get_atom_index(_a)
        self._coords = np.delete(self._coords, ai, axis=0)
        super().del_atom(_a)
