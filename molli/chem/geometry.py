from __future__ import annotations

# === MOLLI IMPORTS ===
from . import (
    Element,
    ElementLike,
    Atom,
    AtomLike,
    Bond,
    Promolecule,
    PromoleculeLike,
    Connectivity,
)
from ..parsing import read_xyz

# =====================
from typing import Any, List, Iterable, Generator, TypeVar, Generic, Callable
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from io import StringIO, BytesIO
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


def _nans(shape, dtype) -> np.ndarray:
    res = np.empty(shape, dtype=dtype)
    res.fill(np.nan)
    return res


class CartesianGeometry(Promolecule):
    """
    Cartesian Geometry Class
    Stores molecular geometry in ANGSTROM floating points.
    This version is generalizable to arbitrary coordinates and data types
    """

    __slots__ = ("_atoms", "_coords", "_name", "_dtype")

    def __init__(
        self,
        n_atoms: int = 0,
        name: str = "unnamed",
        *,
        dtype: str = "float32",
    ):
        # Type of coordinates
        super().__init__(n_atoms=n_atoms, name=name)
        self._dtype = str(dtype)
        self._coords = _nans((self.n_atoms, 3), dtype=self._dtype)

    # ADD METHODS TO OVERRIDE ADDING ATOMS!

    def add_atom(self, a: Atom, coord: ArrayLike):
        _a = super().add_atom(a)
        _coord = np.array(coord, dtype=self._dtype)
        if not _coord.shape == (3,):
            raise ValueError(
                "Inappropriate coordinates for atom (interpreted as {_coord})"
            )
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
        _coords = np.array(other, self._dtype)

        na = self.n_atoms

        match _coords.shape:
            case (x,) if x == na * 3:
                self._coords = np.reshape(_coords, (na, 3))

            case (x, y) if x == na and y == 3:
                self._coords = _coords

            case _:
                raise ValueError(
                    f"Failed to assign array of shape {_coords.shape} to a geometry with {na} atoms."
                )

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
            comment = f"{self.name} [produced with molli]"
            output.write(f"{self.n_atoms}\n{comment}\n")

        for i in range(self.n_atoms):
            s = self.atoms[i].element.symbol
            x, y, z = self.coords[i]
            output.write(f"{s:<5} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    def to_xyzblock(self) -> str:
        strio = StringIO()
        self.dump_xyz(strio)
        return strio.getvalue()

    @classmethod
    def from_xyz(
        cls,
        input: str | StringIO | BytesIO,
        source_units: str = "Angstrom",
    ):
        """This method imports an xyz string and produces a cartesian geometry object"""
        return next(cls.yield_from_xyz(input, source_units))

    @classmethod
    def yield_from_xyz(
        cls,
        input: str | StringIO,
        source_units: str = "Angstrom",
    ):
        match input:
            case str():
                xyzio = StringIO(input)
            case StringIO():
                xyzio = input
            case _:
                xyzio = input

        for xyzblock in read_xyz(xyzio):
            g = cls(xyzblock.n_atoms)
            for i, s in enumerate(xyzblock.symbols):
                g.atoms[i].element = Element[s]
            g.coords = xyzblock.coords

            if DistanceUnit[source_units] != DistanceUnit.Angstrom:
                g.scale(DistanceUnit[source_units].value)

            yield g

    def scale(self, factor: float, allow_inversion=False):
        """
        Simple multiplication of all coordinates by a factor.
        Useful for unit conversion.
        """
        if factor < 0 and not allow_inversion:
            raise ValueError(
                "Scaling with a negative factor can only be performed with explicit `scale(factor, allow_inversion = True)`"
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
        i1, i2 = self.yield_atom_indices((a1, a2))
        return np.linalg.norm(self.coords[i1] - self.coords[i2])

    def distance_to_point(self, a: AtomLike, p: ArrayLike) -> float:
        i = self.index_atom(a)
        return np.linalg.norm(self.coords[i] - p)

    def angle(self, a1: AtomLike, a2: AtomLike, a3: AtomLike) -> float:
        """
        Compute an angle
        """
        i1, i2, i3 = self.yield_atom_indices((a1, a2, a3))

        v1 = self.coords[i1] - self.coords[i2]
        v2 = self.coords[i3] - self.coords[i2]

        # dt = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # return np.arccos(dt)

        return _angle(v1, v2)

    def coord_subset(self, atoms: Iterable[AtomLike]) -> np.ndarray:
        indices = list(self.yield_atom_indices(atoms))
        return self.coords[indices]

    def dihedral(
        self,
        a1: AtomLike,
        a2: AtomLike,
        a3: AtomLike,
        a4: AtomLike,
    ) -> float:
        i1, i2, i3, i4 = self.yield_atom_indices((a1, a2, a3, a4))

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
