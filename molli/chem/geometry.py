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
    Enumerates through commonly used distance units

    """
    A = 1.0
    Angstrom = A
    Bohr = 1.88973
    au = Bohr
    fm = 100_000.0
    pm = 100.0
    nm = 0.100


def _angle(v1: ArrayLike, v2: ArrayLike) -> float:
    """
    Computes the angle between two vectors in radians

    This parameter is communtative 

    :param v1: First vector
    :type v1: ArrayLike
    :param v2: Second vector
    :type v2: ArrayLike
    :return: Angle between vectors in radians
    :rtype: float
    """    
    dt = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(dt)


class CartesianGeometry(Promolecule):
    """
    Stores molecular geometry in ANGSTROM floating points.
    This version is generalizable to arbitrary coordinates and data types
    """
    _coords_dtype = np.float64
    __slots__ = "_atoms", "_coords", "_name", "charge", "mult"

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
            self.coords = coords or np.nan

    # ADD METHODS TO OVERRIDE ADDING ATOMS!

    def add_atom(self, a: Atom, coord: ArrayLike):
        """
        Adds an atom to the geometry
        
        :param a: Atom to be added
        :type a: Atom
        :param coord: Coordinates of the atom
        :type coord: ArrayLike
        :raises ValueError: Inappropriate coordinates for atom (interpreted as {_coord})
        """        
        _a = super().add_atom(a)
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
        """
        Adds a new atom to the geometry and returns it
        

        :param element: Element of the atom, defaults to ...
        :type element: Element, optional
        :param isotope: Isotope of the atom, defaults to None
        :type isotope: int, optional
        :param coord: Coordinates of the atom, defaults to [0, 0, 0]
        :type coord: ArrayLike, optional
        :return: The newly created atom
        :rtype: Atom
        """ 
        _a = Atom(element, isotope, **kwargs)
        self.add_atom(_a, coord)
        return _a

    @property
    def coords(self) -> ArrayLike:
        """
        Returns the coordinates of the geometry
        
        :return: The coordinates of the geometry
        :rtype: ArrayLike
        """        
        return self._coords

    @coords.setter
    def coords(self, other: ArrayLike):
        """
        Sets the coordinates of the geometry
        
        :param other: The coordinates to be set
        :type other: ArrayLike
        """        
        self._coords[:] = other

    @property
    def coords_as_list(self) -> List[float]:
        """
        Returns the coordinates as a list of floats
        
        :return: The coordinates as a list of floats
        :rtype: List[float]
        """        
        return self._coords.flatten().tolist()

    def extend(self, other: CartesianGeometry):
        """
        Extends the current geometry with another geometry
        
        :param other: The geometry to be extended with
        :type other: CartesianGeometry

        :raises NotImplementedError: Not implemented
        """        
        raise NotImplementedError

    def dump_xyz(self, output: StringIO, write_header: bool = True) -> None:
        """# `dump_xyz`
        Dumps the xyz file into the output stream
        
        :param output: Output stream
        :type output: StringIO
        :param write_header: Whether to write the header, defaults to True
        :type write_header: bool, optional
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
        Returns a string representation of the xyz file
        
        :param write_header: Whether to write the header, defaults to True
        :type write_header: bool, optional
        :return: String representation of the xyz file
        :rtype: str
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
        """
        This function loads a *single* xyz file into the current instance
        The input should be a stream or file name/path

        :param cls: The class to load the xyz file into
        :type cls: type[CartesianGeometry]
        :param input: XYZ file to be loaded
        :type input: str | Path | IO
        :param name: Name of the geometry, defaults to None
        :type name: str, optional
        :param source_units: Source units should be one of: A == Angstrom, Bohr == au, fm, pm, nm, defaults to "Angstrom"
        :type source_units: str, optional
        :return: The loaded geometry
        :rtype: CartesianGeometry
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
        """
        This function loads a *single* xyz file into the current instance
        The input should be a string instance

        :param cls: The class to load the xyz file into
        :type cls: type[CartesianGeometry]
        :param input: XYZ file to be loaded
        :type input: str
        :param name: Name of the geometry, defaults to None
        :type name: str, optional
        :param source_units: Source units should be one of: A == Angstrom, Bohr == au, fm, pm, nm, defaults to "Angstrom"
        :type source_units: str, optional
        :return: The loaded geometry
        :rtype: CartesianGeometry
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
        """
        This function loads all xyz files from the input stream
        
        :param cls: The class to load the xyz file into
        :type cls: type[CartesianGeometry]
        :param input: Input must be an IOBase instance (typically, an open file)
        :type input: str | Path | IO
        :param name: Name of the geometry, defaults to None
        :type name: str, optional
        :param source_units: Source units should be one of: A == Angstrom, Bohr == au, fm, pm, nm, defaults to "Angstrom"
        :type source_units: str, optional
        :return: The loaded geometry
        :rtype: CartesianGeometry
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
        input: str | Path | IO,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> List[CartesianGeometry]:
        """
        This function loads all xyz files from the input string
        
        :param cls: The class to load the xyz file into
        :type cls: type[CartesianGeometry]
        :param input: Input must be an IOBase instance (typically, an open file)
        :type input: str | Path | IO
        :param name: Name of the geometry, defaults to None
        :type name: str, optional
        :param source_units: Source units should be one of: A == Angstrom, Bohr == au, fm, pm, nm, defaults to "Angstrom"
        :type source_units: str, optional
        :return: The loaded geometry
        :rtype: CartesianGeometry
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
        """
        This function loads all xyz files from the input string
        
        :param cls: The class to load the xyz file into
        :type cls: type[CartesianGeometry]
        :param stream: Input must be an StringIO instance
        :type stream: StringIO
        :param name: Name of the geometry, defaults to None
        :type name: str, optional
        :param source_units: Source units should be one of: A == Angstrom, Bohr == au, fm, pm, nm, defaults to "Angstrom"
        :type source_units: str, optional
        :yield: A loaded geometry
        :rtype: Generator[CartesianGeometry, None, None]
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
        """
        Multiplication of all coordinates by a factor.
        
        :param factor: Scaling factor
        :type factor: float
        :param allow_inversion: If True, the geometry can be inverted, defaults to False
        :type allow_inversion: bool, optional
        :raises ValueError: If the factor is 0
        :raises ValueError: If the factor is negative and inversion is not allowed
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
        """
        Inverts the coordinates with respect to the origin. This also inverts the absolute stereochemistry"
        """        
        self.scale(-1, allow_inversion=True)

    def distance(self, a1: AtomLike, a2: AtomLike) -> float:
        """
        Calculates the distance between two atoms
        
        :param a1: First atom
        :type a1: AtomLike
        :param a2: Second atom
        :type a2: AtomLike
        :return: Distance between the two atoms
        :rtype: float
        """        
        i1, i2 = map(self.get_atom_index, (a1, a2))
        return np.linalg.norm(self.coords[i1] - self.coords[i2])

    def get_atom_coord(self, _a: AtomLike) ->np.ndarray:
        """
        Returns the coordinates of the atom
        
        :param _a: Atom of interest
        :type _a: AtomLike
        :return: Coordinates of the atom
        :rtype: np.ndarray
        """        
        return self.coords[self.get_atom_index(_a)]

    def vector(self, a1: AtomLike, a2: AtomLike | np.ndarray) -> np.ndarray:
        """
        Returns the vector between two atoms
        
        :param a1: First atom
        :type a1: AtomLike
        :param a2: Second atom
        :type a2: AtomLike | np.ndarray
        :return: Vector between the two atoms
        :rtype: np.ndarray
        """        
        v1 = self.get_atom_coord(a1)
        v2 = self.get_atom_coord(a2) if isinstance(a2, AtomLike) else np.array(a2)

        return v2 - v1

    def distance_to_point(self, a: AtomLike, p: ArrayLike) -> float:
        """
        Compute the distance between an atom and a point
        
        :param a: Atom of interest
        :type a: AtomLike
        :param p: Point of interest
        :type p: ArrayLike
        :return: Distance between the atom and the point
        :rtype: float
        """        
        return np.linalg.norm(self.vector(a, p))

    def angle(self, a1: AtomLike, a2: AtomLike, a3: AtomLike) -> float:
        """
        Compute an angle between three atoms in radians
        
        :param a1: First atom
        :type a1: AtomLike
        :param a2: Second atom
        :type a2: AtomLike
        :param a3: Third atom
        :type a3: AtomLike
        :return: Angle between the three atoms in radians
        :rtype: float
        """       
        i1, i2, i3 = map(self.get_atom_index, (a1, a2, a3))

        v1 = self.coords[i1] - self.coords[i2]
        v2 = self.coords[i3] - self.coords[i2]

        # dt = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # return np.arccos(dt)

        return _angle(v1, v2)

    def coord_subset(self, atoms: Iterable[AtomLike]) -> np.ndarray:
        """
        Returns the coordinates of a subset of atoms
        
        :param atoms: Subset of atoms
        :type atoms: Iterable[AtomLike]
        :return: Coordinates of the subset of atoms
        :rtype: np.ndarray
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
        """
        Compute the dihedral angle between four atoms in radians
        
        :param a1: First atom
        :type a1: AtomLike
        :param a2: Second atom
        :type a2: AtomLike
        :param a3: Third atom
        :type a3: AtomLike
        :param a4: Fourth atom
        :type a4: AtomLike
        :return: Dihedral angle between the four atoms in radians
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

    def translate(self, vector: ArrayLike):
        """
        Translate coordinates
        
        :param vector: Translation vector
        :type vector: ArrayLike
        """        
        v = np.array(vector)
        self.coords += v[np.newaxis, :]

    def centroid(self) -> np.ndarray:
        """
        Centroid of the molecule
        
        :return: Centroid of the molecule
        :rtype: np.ndarray
        """        
        return np.average(self.coords, axis=0)
    

    def rmsd(self, other: CartesianGeometry, validate_elements=True):
        """
        Compute the RMSD between two geometries
        
        :param other: Other geometry
        :type other: CartesianGeometry
        :param validate_elements: Whether to validate the elements of the two geometries, defaults to True
        :type validate_elements: bool, optional
        :raises ValueError: If the geometries have different number of atoms
        :raises ValueError: If the geometries have different elements
        :raises NotImplementedError: If the geometries have different number of atoms
    
        """        
        if other.n_atoms != self.n_atoms:
            raise ValueError("Cannot compare geometries with different number of atoms")

        if validate_elements == True and self.elements != other.elements:
            raise ValueError(
                "Cannot compare two molecules with different lists of elements"
            )

        raise NotImplementedError

    def transform(self, _t_matrix: ArrayLike, /, validate=False):
        """
        Transform the coordinates of the molecule

        :param _t_matrix: Transformation matrix
        :type _t_matrix: ArrayLike
        :param validate: Whether to validate the transformation matrix, defaults to False
        """        
        t_matrix = np.array(_t_matrix)
        self.coords = t_matrix @ self.coords

    def del_atom(self, _a: AtomLike):
        """
        Delete an atom from the molecule

        :param _a: Atom to delete
        :type _a: AtomLike
        """        
        ai = self.index_atom(_a)
        self._coords = np.delete(self._coords, ai, 0)
        super().del_atom(_a)