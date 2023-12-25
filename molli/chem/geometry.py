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
# `molli.chem.geometry`

This submodule defines classes `CartesianGeometry`.
"""

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
from typing import Any, List, Iterable, Generator, TypeVar, Generic, Callable, IO
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from io import StringIO, BytesIO, IOBase
from functools import wraps
from pathlib import Path
import math


class DistanceUnit(Enum):
    """This is an Enumeration class for assigning commonly used distance units

    Parameters
    ----------
    Enum :
        Accepts integer enumerations for different distance units

    Examples
    -------
        >>> ml.DistanceUnit(1.0) == ml.DistanceUnit.A
        True
    """

    A = 1.0
    Angstrom = A
    Bohr = 1.88973
    au = Bohr
    fm = 100_000.0
    pm = 100.0
    nm = 0.100


def _angle(v1: ArrayLike, v2: ArrayLike) -> float:
    dt = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(dt)


class CartesianGeometry(Promolecule):
    """Stores molecular geometry in ANGSTROM floating points.
    This version is generalizable to arbitrary coordinates and data types.
    This is a parent class that employs methods that work on a Promolecule
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
            self.coords = coords if coords is not None else other.coords
        else:
            self.coords = coords if coords is not None else np.nan

    # ADD METHODS TO OVERRIDE ADDING ATOMS!

    def add_atom(self, a: Atom, coord: ArrayLike) -> None:
        """Adds atom to CartesianGeometry

        Parameters
        ----------
        a : Atom
            Atom to add
        coord : ArrayLike
            Coordinates to add

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
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.add_atom(new_atom, [0,0,0])
            >>> cargeom.n_atoms
        """

        super().append_atom(a)
        _coord = np.array(coord, dtype=self._coords_dtype)
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
        """Adds a new atom and returns it

        Parameters
        ----------
        element : Element, optional
            Element of the atom, by default ...
        isotope : int, optional
            Isotope of the atom, by default None
        coord : ArrayLike, optional
            Coordinates of the atom, by default [0, 0, 0]

        Returns
        -------
        Atom
            Returns Atom instance

        Examples
        -------
        The Molecule class inherits new_atom()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.n_atoms
            44
            >>> dendrobine.new_atom(element="C", isotope=None, coord=[0,0,0])
            Atom(element=C, isotope=None, ...)
            >>> dendrobine.n_atoms
            45
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.new_atom(element="C", isotope=None, coord=[0,0,0])
            44
            >>> cartgeom.new_atom(element="C", isotope=None, coord=[0,0,0])
            Atom(element=C, isotope=None, ...)
            >>> cartgeom.n_atoms
            45

        """

        _a = Atom(element, isotope, **kwargs)
        self.add_atom(_a, coord)
        return _a

    @property
    def coords(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Coordinates of Geometry

        Examples
        -------
            >>> geom = ml.CartesianGeometry()
            >>> geom.add_atom(ml.Atom("C"), [0,0,0])
            >>> geom.coords
            array([[0., 0., 0.]])
        """

        return self._coords

    @coords.setter
    def coords(self, other: ArrayLike):
        self._coords[:] = other

    @property
    def coords_as_list(self) -> List[float]:
        """
        Returns
        -------
        List[float]
            Returns flattened coordinates of CartesianGeometry as a list

        Examples
        -------
            >>> geom = ml.CartesianGeometry()
            >>> geom.add_atom(ml.Atom("C"), [0,0,0])
            >>> geom.coords_as_list
            [0.0, 0.0, 0.0]
        """

        return self._coords.flatten().tolist()

    def extend(self, other: Iterable[CartesianGeometry]):
        """Currently Not Implemented

        Parameters
        ----------
        other : Iterable[CartesianGeometry]
            iterable to extend cartesiangeometry from other cartesian geometries

        """
        raise NotImplementedError

    def dump_xyz(
        self, output: StringIO, write_header: bool = True, *, fmt: str = "12.6f"
    ) -> None:
        """Dumps the xyz file into the output stream

        Parameters
        ----------
        output : StringIO
            Output stream
        write_header : bool, optional
            Whether to write the header, by default True
        fmt : str, optional
            Format of the string, by default "12.6f"

        Examples
        -------
            >>> geom = ml.CartesianGeometry()
            >>> geom.add_atom(ml.Atom("C"), [0,0,0])
            >>> with open('test.xyz', 'w') as f:
            >>>     geom.dump_xyz(f)
            1
            unknown
            C         0.000000     0.000000     0.000000

        """

        # Header need not be written in certain files
        # Like ORCA inputs
        if write_header:
            if hasattr(self, "name"):
                comment = f"{self.name}"
            else:
                comment = f"{type(self)}"
            output.write(f"{self.n_atoms}\n{comment}\n")

        for i in range(self.n_atoms):
            s = self.atoms[i].element.symbol
            x, y, z = self.coords[i]
            output.write(f"{s:<5} {x:{fmt}} {y:{fmt}} {z:{fmt}}\n")

    def dumps_xyz(self, write_header: bool = True) -> str:
        """Dumps the xyz file into the output stream

        Parameters
        ----------
        write_header : bool, optional
            Whether to write the header, by default True

        Returns
        -------
        str
            The xyz block

        Examples
        -------
            >>> geom = ml.CartesianGeometry()
            >>> geom.add_atom(ml.Atom("C"), [0,0,0])
            >>> geom.dumps_xyz()
            1
            unknown
            C         0.000000     0.000000     0.000000
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
        """This function loads a single xyz file into the current instance.
        This can be a stream, filepath, or string.

        Parameters
        ----------
        cls : type[CartesianGeometry]
            The class to load the xyz file into
        input : str | Path | IO
            XYZ file to be loaded
        name : str, optional
            Name of the geometry, by default None
        source_units : str, optional
            Units to use when reading, by default "Angstrom"

        Returns
        -------
        CartesianGeometry
            Returns CartesianGeometry

        Examples
        -------
        The Molecule class inherits load_xyz()
            >>> ml.Molecule.load_xyz(ml.files.dendrobine_xyz, name='dendrobine')
            Molecule(name='dendrobine', formula='C16 H25 N1 O2')
        If desired, one can work directly with CartesianGeometry class instead
            >>> ml.CartesianGeometry.load_xyz(ml.files.dendrobine_xyz,
            name='dendrobine')
            CartesianGeometry(name='dendrobine', formula='C16 H25 N1 O2')
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
        """This function loads a single xyz file into the current instance.
        This can only be a string

        Parameters
        ----------
        cls : type[CartesianGeometry]
            The class to load the xyz file into
        input : str
            XYZ block as a string
        name : str, optional
            Name of the geometry, by default None
        source_units : str, optional
            Units to use when reading, by default "Angstrom"

        Returns
        -------
        CartesianGeometry
            Returns CartesianGeometry

        Examples
        -------
        The Molecule class inherits loads_xyz()
            >>> with open(ml.files.dendrobine_xyz) as f:
            >>>     ml.Molecule.loads_xyz(f.read(), name='dendrobine')
            Molecule(name='dendrobine', formula='C16 H25 N1 O2')
        If desired, one can work directly with CartesianGeometry class instead
            >>> with open(ml.files.dendrobine_xyz) as f:
            >>>     ml.CartesianGeometry.loads_xyz(f.read(), name='dendrobine')
            CartesianGeometry(name='dendrobine', formula='C16 H25 N1 O2')
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
        """This function loads all xyz files from the input

        Parameters
        ----------
        cls : type[CartesianGeometry]
            The class to load the xyz file into
        input : str | Path | IO
            XYZ file to be loaded
        name : str, optional
            Name of the geometry, by default None
        source_units : str, optional
            Units to use when reading, by default "Angstrom"

        Returns
        -------
        List[CartesianGeometry]
            Returns list of cartesian geometries

        Examples
        -------
        The Molecule class inherits load_all_xyz()
            >>> ml.Molecule.load_all_xyz(ml.files.pentane_confs_xyz)
            [Molecule(name='unnamed', formula='C5 H12'), ...]
        If desired, one can work directly with CartesianGeometry class instead
            >>> ml.CartesianGeometry.load_all_xyz(ml.files.pentane_confs_xyz)
            [CartesianGeometry(name='unnamed', formula='C5 H12'), ...]
        """

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
        input: str,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> List[CartesianGeometry]:
        """This function loads all xyz files from the input string

        Parameters
        ----------
        cls : type[CartesianGeometry]
            The class to load the xyz file into
        input : str
            XYZ block as a string
        name : str, optional
            Name of the geometry, by default None
        source_units : str, optional
            Units to use when reading, by default "Angstrom"

        Returns
        -------
        List[CartesianGeometry]
            Returns list of cartesian geometries

        Examples
        -------
        The Molecule class inherits loads_all_xyz()
            >>> with open(ml.files.pentane_confs_xyz) as f:
            >>>     ml.Molecule.loads_all_xyz(f.read())
            [Molecule(name='dendrobine', formula='C16 H25 N1 O2'),...]
        If desired, one can work directly with CartesianGeometry class instead
            >>> with open(ml.files.pentane_confs_xyz) as f:
            >>>     ml.CartesianGeometry.loads_all_xyz(f.read())
            [CartesianGeometry(name='dendrobine', formula='C16 H25 N1 O2'),...]
        """

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
    ) -> Generator[CartesianGeometry, None, None]:
        """Yields generator of CartesianGeometry from stream

        Parameters
        ----------
        cls : type[CartesianGeometry]
            The class to load the xyz file into
        stream : StringIO
            Stream to read from
        name : str, optional
            Name of the geometry, by default None
        source_units : str, optional
            Units to use when reading, by default "Angstrom"

        Yields
        ------
        Generator[CartesianGeometry, None, None]
            Yields Generator of CartesianGeometry

        Examples
        -------
        The Molecule class inherits yield_from_xyz()
            >>> with open(ml.files.dendrobine_xyz) as f:
            >>>     ml.Molecule.yield_from_xyz(f, name='dendrobine')
            <generator object CartesianGeometry.yield_from_xyz at ...>
        If desired, one can work directly with CartesianGeometry class instead
            >>> with open(ml.files.dendrobine_xyz) as f:
            >>>     ml.CartesianGeometry.yield_from_xyz(f, name='dendrobine')
            <generator object CartesianGeometry.yield_from_xyz at ...>
        """

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

    def scale(self, factor: float, allow_inversion=False) -> None:
        """Multiplies all coordinates by a factor

        Parameters
        ----------
        factor : float
            Factor to scale by
        allow_inversion : bool, optional
            Allows inversion of coordinates, by default False

        Examples
        -------
        The Molecule class inherits scale()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.coords
            array([[ 1.2960e+00 -2.3190e-01  1.2670e+00],...
            >>> dendrobine.scale(0.5)
            >>> dendrobine.coords
            array([[ 6.48000e-01, -1.15950e-01,  6.33500e-01],...
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.scale(0.5)
            >>> cartgeom.coords
            array([[ 6.48000e-01, -1.15950e-01,  6.33500e-01],...
        """

        if factor < 0 and not allow_inversion:
            raise ValueError(
                "Scaling with a negative factor can only be performed with"
                " explicit `scale(factor, allow_inversion = True)`"
            )

        if factor == 0:
            raise ValueError("Scaling with a factor == 0 is not allowed.")

        self.coords *= factor

    def invert(self) -> None:
        """Coordinates are inverted wrt the origin. This also inverts
        inverts the absolute stereochemistry

        Examples
        -------
        The Molecule class inherits invert()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.coords
            array([[ 1.2960e+00 -2.3190e-01  1.2670e+00],...
            >>> dendrobine.invert()
            >>> dendrobine.coords
            array([[-1.2960e+00,  2.3190e-01, -1.2670e+00],
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.invert()
            >>> cartgeom.coords
            array([[-1.2960e+00,  2.3190e-01, -1.2670e+00],
        """

        self.scale(-1, allow_inversion=True)

    def distance(self, a1: AtomLike, a2: AtomLike) -> float:
        """Calculates the distance between two atoms

        Parameters
        ----------
        a1 : AtomLike
            First atom
        a2 : AtomLike
            Second atom

        Returns
        -------
        float
            Distance between the two atoms

        Examples
        -------
        The Molecule class inherits distance()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.distance(0,1)
            1.520002194077364
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.distance(0,1)
            1.520002194077364
        """

        i1, i2 = map(self.get_atom_index, (a1, a2))
        return np.linalg.norm(self.coords[i1] - self.coords[i2])

    def get_atom_coord(self, _a: AtomLike) -> np.ndarray:
        """Returns the coordinates of the atom

        Parameters
        ----------
        _a : AtomLike
            Atom of interest

        Returns
        -------
        np.ndarray
            Coordinates of the atom

        Examples
        -------
        The Molecule class inherits get_atom_coord()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.get_atom_coord(0)
            array([ 1.296 , -0.2319,  1.267 ])
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.get_atom_coord(0,1)
            array([ 1.296 , -0.2319,  1.267 ])
        """

        return self.coords[self.get_atom_index(_a)]

    def vector(self, a1: AtomLike, a2: AtomLike | np.ndarray) -> np.ndarray:
        """Returns the vector between two atoms or one atom and array

        Parameters
        ----------
        a1 : AtomLike
            First atom
        a2 : AtomLike | np.ndarray
            Second atom or array

        Returns
        -------
        np.ndarray
            Vector between the two atoms

        Examples
        -------
        The Molecule class inherits vector()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.vector(0,1)
            array([-1.2387,  0.2093,  0.8557])
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.vector(0,1)
            array([-1.2387,  0.2093,  0.8557])
        """

        v1 = self.get_atom_coord(a1)
        v2 = self.get_atom_coord(a2) if isinstance(a2, AtomLike) else np.array(a2)

        return v2 - v1

    def distance_to_point(self, a: AtomLike, p: ArrayLike) -> float:
        """Compute the distance between an atom and a point

        Parameters
        ----------
        a : AtomLike
            Atom of interest
        p : ArrayLike
            Point of interest

        Returns
        -------
        float
            Distance between the atom and the point

        Examples
        -------
        The Molecule class inherits distance_to_point()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.distance_to_point(0,[0,0,0])
            1.827206230834385
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.distance_to_point(0,[0,0,0])
            1.827206230834385
        """

        return np.linalg.norm(self.vector(a, p))

    def angle(self, a1: AtomLike, a2: AtomLike, a3: AtomLike) -> float:
        """Compute an angle between three atoms in radians

        Parameters
        ----------
        a1 : AtomLike
            First atom
        a2 : AtomLike
            Second atom
        a3 : AtomLike
            Third atom

        Returns
        -------
        float
            Angle between the three atoms in radians

        Examples
        -------
        The Molecule class inherits angle()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.angle(0,1,2)
            1.8082869758837117
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.angle(0,1,2)
            1.8082869758837117
        """

        i1, i2, i3 = map(self.get_atom_index, (a1, a2, a3))

        v1 = self.coords[i1] - self.coords[i2]
        v2 = self.coords[i3] - self.coords[i2]

        # dt = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # return np.arccos(dt)

        return _angle(v1, v2)

    def coord_subset(self, atoms: Iterable[AtomLike]) -> np.ndarray:
        """Returns the coordinates of a subset of atoms

        Parameters
        ----------
        atoms : Iterable[AtomLike]
            Subset of atoms

        Returns
        -------
        np.ndarray
            Coordinates of the subset of atoms

        Examples
        -------
        The Molecule class inherits coord_subset()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.coord_subset([0,1,2])
            array([[ 1.296 , -0.2319,  1.267 ],
            [ 0.0573, -0.0226,  2.1227],
            [-1.0974, -0.4738,  1.2059]])
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.coord_subset([0,1,2])
            array([[ 1.296 , -0.2319,  1.267 ],
            [ 0.0573, -0.0226,  2.1227],
            [-1.0974, -0.4738,  1.2059]])
        """

        indices = list(map(self.get_atom_index, atoms))
        return self.coords[indices]

    def dihedral(
        self,
        a1: AtomLike,
        a2: AtomLike,
        a3: AtomLike,
        a4: AtomLike,
    ) -> float:
        """Compute the dihedral angle between four atoms in radians

        Parameters
        ----------
        a1 : AtomLike
            First atom
        a2 : AtomLike
            Second atom
        a3 : AtomLike
            Third atom
        a4 : AtomLike
            Fourth atom

        Returns
        -------
        float
            Dihedral angle between the four atoms in radians

        Examples
        -------
        The Molecule class inherits dihedral()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.dihedral(0,1,2,3)
            -0.3286236550063439
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.dihedral(0,1,2,3)
            -0.3286236550063439
        """

        i1, i2, i3, i4 = map(self.get_atom_index, (a1, a2, a3, a4))

        u1 = self.coords[i2] - self.coords[i1]
        u2 = self.coords[i3] - self.coords[i2]
        u3 = self.coords[i4] - self.coords[i3]

        u2xu3 = np.cross(u2, u3)
        u1xu2 = np.cross(u1, u2)
        arg1 = np.linalg.norm(u2) * np.dot(u1, u2xu3)
        arg2 = np.dot(u1xu2, u2xu3)

        return np.arctan2(arg1, arg2)

    def translate(self, vector: ArrayLike) -> None:
        """Translates coordinates inplace

        Parameters
        ----------
        vector : ArrayLike
            Translation vector

        Examples
        -------
        The Molecule class inherits translate()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.coords
            array([[ 1.2960e+00, -2.3190e-01,  1.2670e+00],...
            >>> dendrobine.translate([1,1,1])
            >>> dendrobine.coords
            array([[ 2.296 ,  0.7681,  2.267 ],...
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.translate([1,1,1])
            >>> cartgeom.coords
            array([[ 2.296 ,  0.7681,  2.267 ],...
        """

        v = np.array(vector)
        self.coords += v[np.newaxis, :]

    def centroid(self) -> np.ndarray:
        """Centroid of the molecule

        Returns
        -------
        np.ndarray
            Centroid of the molecule

        Examples
        -------
        The Molecule class inherits centroid()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.centroid()
            array([ 0.16483864, -0.16130455, -1.00852727])
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.centroid()
            array([ 0.16483864, -0.16130455, -1.00852727])
        """

        return np.average(self.coords, axis=0)

    def rmsd(self, other: CartesianGeometry, validate_elements=True):
        """Currently Not Implemented

        Parameters
        ----------
        other : CartesianGeometry
            Other CartesianGeometry for comparison
        validate_elements : bool, optional
            Validates the elements between the two geometries are
            equal, by default True

        """
        raise NotImplementedError("RMSD Calculation Currently Not Implemented")

        if other.n_atoms != self.n_atoms:
            raise ValueError("Cannot compare geometries with different number of atoms")

        if validate_elements == True and self.elements != other.elements:
            raise ValueError(
                "Cannot compare two molecules with different lists of elements"
            )

    def transform(self, _t_matrix: ArrayLike, /, validate=False) -> None:
        """Transform the coordinates of the molecule

        Parameters
        ----------
        _t_matrix : ArrayLike
            Transformation matrix
        validate : bool, optional
            Whether to validate the transformation matrix, by default False
            Currently Not Implemented

        Examples
        -------
        The Molecule class inherits transform()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.coords
            array([[ 1.2960e+00, -2.3190e-01,  1.2670e+00],...
            >>> t_matrix = ml.math.rotation_matrix_from_axis([0,0,1],90)
            >>> dendrobine.transform(t_matrix)
            array([[-7.88021233e-01, -1.05471140e+00,  1.26700000e+00],...
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.transform(t_matrix)
            array([[-7.88021233e-01, -1.05471140e+00,  1.26700000e+00],...
        """

        t_matrix = np.array(_t_matrix)
        self.coords = self.coords @ t_matrix

    def del_atom(self, _a: AtomLike):
        """Deletes an atom from the CartesianGeometry

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
        If desired, one can work directly with CartesianGeometry class instead
            >>> cartgeom = ml.CartesianGeometry(dendrobine)
            >>> cartgeom.del_atom(0)
            >>> cartgeom.get_atom(0)
            Atom(element=C, isotope=None, label='C', formal_charge=0, formal_spin=0)
        """
        ai = self.get_atom_index(_a)
        self._coords = np.delete(self._coords, ai, axis=0)
        super().del_atom(_a)
