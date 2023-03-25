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
    # `DistanceUnit`
    
    Enumerates through commonly used distance units

    Args:

        DistanceUnit(Enum): inherited class for enumeration 

    Potential Uses:
    ```Python
        nano_meter = DistanceUnit(0.100) # nanometer 
        nano_meter == DistanceUnit.nm  # True 
    ```
    """
    A = 1.0
    Angstrom = A
    Bohr = 1.88973
    au = Bohr
    fm = 100_000.0
    pm = 100.0
    nm = 0.100


def _angle(v1: ArrayLike, v2: ArrayLike) -> float:
    """# `_angle`
    Computes the angle between two vectors in degrees
    This parameter is communtative 
    
    
    
    ## Parameters
    
    `v1: ArrayLike`
        First vector of interest
    `v2: ArrayLike`
        Second vector of interest
    
    ## Returns
    
    `float`
        the angle between the vectors in degrees 

    ## Examples
        ``` Python
        _angle([1, 0, 0], [0, 1, 0])
        90.0
        ```
    """    
    dt = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(dt)


class CartesianGeometry(Promolecule):
    """
    Cartesian Geometry Class
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
        """# `add_atom`
        Adds an atom to the geometry
        
        
        
        ## Parameters
        
        `a: Atom`

        `coord: ArrayLike`
            The coordinates of the atom
        
        ## Raises
        
        `ValueError`
            If the coordinates are not of shape (3,)
        
        ## Examples
            ``` Python
            geom.add_atom(Atom(Element.H), coord=[0, 0, 0])
            ```
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
        """# `new_atom`
        Adds a new atom to the geometry and returns it
        
        
        
        ## Parameters
        
        `element: Element`, optional, default: `...`
            
        `isotope: int`, optional, default: `None`
            _description_
        `coord: ArrayLike`, optional, default: `[0, 0, 0]`
            _description_
        
        ## Returns
        
        `Atom`
            The newly created atom
        
        ## Examples
            ``` Python
            geom.new_atom(Element.H, coord=[0, 0, 0])
            ```
        """        
        _a = Atom(element, isotope, **kwargs)
        self.add_atom(_a, coord)
        return _a

    @property
    def coords(self) -> ArrayLike:
        """# `coords`
        Returns the coordinates of the geometry
        
        
        
        ## Returns
        
        `ArrayLike`
            The coordinates of the geometry
        
        ## Examples
            ``` Python
            geom.coords
            array([[0., 0., 0.],
                     [1., 1., 1.]])
            ```
        """        
        return self._coords

    @coords.setter
    def coords(self, other: ArrayLike):
        """# `coords`
        Sets the coordinates of the geometry
        
        
        
        ## Parameters
        
        `other: ArrayLike`
            _coords to be set to
        """        
        self._coords[:] = other

    @property
    def coords_as_list(self) -> List[float]:
        """# `coords_as_list`
        Returns the coordinates as a list of floats
        
        
        
        ## Returns
        
        `List[float]`
            The coordinates as a list of floats
            
        
        ## Examples
            ``` Python
            geom.coords_as_list
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

            ```
        """        
        return self._coords.flatten().tolist()

    def extend(self, other: CartesianGeometry):
        """# `extend`
        Extends the current geometry with another geometry
        
        
        
        ## Parameters
        
        `other: CartesianGeometry`
            The geometry to be extended with
        
        ## Raises
        
        `NotImplementedError`
            If the other geometry is not a CartesianGeometry
        """        
        raise NotImplementedError

    def dump_xyz(self, output: StringIO, write_header: bool = True) -> None:
        """# `dump_xyz`
        Dumps the xyz file into the output stream
        
        
        
        ## Parameters
        
        `output: StringIO`
            The output stream to write to
        `write_header: bool`, optional, default: `True`
            Whether to write the header of the xyz file
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
        """# `dumps_xyz`
        Returns a string representation of the xyz file
        
        
        
        ## Parameters
        
        `write_header: bool`, optional, default: `True`
            Whether to write the header of the xyz file
        
        ## Returns
        
        `str`
            String representation of the xyz file
        
        ## Examples
            ``` Python
            mol = CartesianGeometry.from_smiles("C")
            mol.dumps_xyz()
            '1
            ```
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
            The class to load the xyz file into

        `input: str | Path | IO`
            Input must be an IOBase instance (typically, an open file)
            Alternatively, a string or path will be considered as a path to file that will need to be opened.
        `source_units: str`, optional, default: `"Angstrom"`
            Source units should be one of: A == Angstrom, Bohr == au, fm, pm, nm
        
        ## Returns

        `CartesianGeometry`
            The loaded geometry
        
        
        ## Examples
            ``` Python
            mol = CartesianGeometry.load_xyz("mol.xyz")
            ```
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
            instance to load the xyz file into

        `input: str`
            a string or path will be considered as a path to file that will need to be opened.
        `name: str`, optional, default: `None`
            Name of the molecule
        `source_units: str`, optional, default: `"Angstrom"`
            Source units should be one of: A == Angstrom, Bohr == au, fm, pm, nm
        
        ## Returns

        `CartesianGeometry`
            The loaded geometry

        ## Examples
            ``` Python
            mol = CartesianGeometry.loads_xyz("mol.xyz")
            ```
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
        """# `load_all_xyz`
        This function loads all xyz files from the input stream
        
        
        
        ## Parameters
        
        `cls: type[CartesianGeometry]`
            The class to load the xyz file into
        `input: str | Path | IO`
            Input must be an IOBase instance (typically, an open file)
        `name: str`, optional, default: `None`
            Name of the molecule
        `source_units: str`, optional, default: `"Angstrom"`
            Source units should be one of: A == Angstrom, Bohr == au, fm, pm, nm
        
        ## Returns
        
        `List[CartesianGeometry]`
            A list of loaded geometries
        
        ## Examples
            ``` Python
            mol = CartesianGeometry.load_all_xyz("mol.xyz")
            ```
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
        """# `loads_all_xyz`
        This function loads all xyz files from the input string
        
        
        
        ## Parameters
        
        `cls: type[CartesianGeometry]`
            The class to load the xyz file into
        `input: str | Path | IO`
            Input must be an IOBase instance (typically, an open file)
            Alternatively, a string or path will be considered as a path to file that will need to be opened.
        `name: str`, optional, default: `None`
            Name of the molecule
        `source_units: str`, optional, default: `"Angstrom"`
            Source units should be one of: A == Angstrom, Bohr == au, fm, pm, nm
        
        ## Returns
        
        `List[CartesianGeometry]`
            A list of loaded geometries
        
        ## Examples
            ``` Python
            mol = CartesianGeometry.loads_all_xyz("mol.xyz")
            ```
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
    ):
        """# `yield_from_xyz`
        This function loads all xyz files from the input string
        
        
        
        ## Parameters
        
        `cls: type[CartesianGeometry]`
            The class to load the xyz file into
        `stream: StringIO`
            Input must be an StringIO instance 
        `name: str`, optional, default: `None`
            Name of the molecule
        `source_units: str`, optional, default: `"Angstrom"`
            Source units should be one of: A == Angstrom, Bohr == au, fm, pm, nm
        
        ## Yields
        
        `CartesianGeometry`
            A loaded geometry
        
        ## Examples
            ``` Python
            mol = CartesianGeometry.yield_from_xyz("mol.xyz")
            ```
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
        """# `scale`
        Multiplication of all coordinates by a factor.
        
        
        
        ## Parameters
        
        `factor: float`
            Factor to Multiply by
        `allow_inversion: bool`, optional, default: `False`
            Allow inversion of the geometry
        
        ## Raises
        
        `ValueError`
            Scaling with a negative factor can only be performed with explicit `scale(factor, allow_inversion = True)`
        `ValueError`
            Scaling with a factor == 0 is not allowed.
        ## Examples
            ``` Python
            mol.scale(1.2)
            ```
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
        """# `invert`
        Inverts the coordinates with respect to the origin. This also inverts the absolute stereochemistry"
        """        
        self.scale(-1, allow_inversion=True)

    def distance(self, a1: AtomLike, a2: AtomLike) -> float:
        """# `distance`
        Calculates the distance between two atoms
        
        
        
        ## Parameters
        
        `a1: AtomLike`
            First atom of interest
        `a2: AtomLike`
            Second atom of interest
        
        ## Returns
        
        `float`
            Distance between the two atoms
        
        ## Examples
            ``` Python
            mol.distance(1, 2)
            mol.distance("H", "O")
            ```
        """        
        i1, i2 = map(self.get_atom_index, (a1, a2))
        return np.linalg.norm(self.coords[i1] - self.coords[i2])

    def get_atom_coord(self, _a: AtomLike) ->np.ndarray:
        """# `get_atom_coord`
        Returns the coordinates of the atom
        
        
        
        ## Parameters
        
        `_a: AtomLike`
            The atom of interest
        
        ## Returns
        
        `np.ndarray`
            The coordinates of the atom
        
        ## Examples
            ``` Python
            mol.get_atom_coord(1)
            mol.get_atom_coord("H")
            ```
        """        
        return self.coords[self.get_atom_index(_a)]

    def vector(self, a1: AtomLike, a2: AtomLike | np.ndarray) -> np.ndarray:
        """# `vector`
        Returns the vector between two atoms
        
        
        
        ## Parameters
        
        `a1: AtomLike`
            First atom of interest
        `a2: AtomLike | np.ndarray`
            Second atom of interest
            Alternatively, a numpy array can be passed to calculate the vector between the first atom and the array
        
        ## Returns
        
        `np.ndarray`
            The vector between the two atoms
        
        ## Examples
            ``` Python
            mol.vector(1, 2)
            mol.vector("H", "O")
            mol.vector("H", "O") = [1, 2, 3]
            ```
        """        
        v1 = self.get_atom_coord(a1)
        v2 = self.get_atom_coord(a2) if isinstance(a2, AtomLike) else np.array(a2)

        return v2 - v1

    def distance_to_point(self, a: AtomLike, p: ArrayLike) -> float:
        """# `distance_to_point`
        Compute the distance between an atom and a point
        
        
        
        ## Parameters
        
        `a: AtomLike`
            Atom of interest
        `p: ArrayLike`
            Point of interest
        
        ## Returns
        
        `float`
            Distance between the atom and the point
        
        ## Examples
            ``` Python
            mol.distance_to_point(1, [1, 2, 3])
            mol.distance_to_point("H", [1, 2, 3]) = 1.2
            ```
        """        
        return np.linalg.norm(self.vector(a, p))

    def angle(self, a1: AtomLike, a2: AtomLike, a3: AtomLike) -> float:
        """# `angle`
        Compute an angle between three atoms in degrees
        
        
        
        ## Parameters
        
        `a1: AtomLike`
            First atom of interest
        `a2: AtomLike`
            Second atom of interest
        `a3: AtomLike`
            Third atom of interest
        
        ## Returns
        
        `float`
            Angle between the three atoms in degrees
        
        ## Examples
            ``` Python
            mol.angle(1, 2, 3)
            ```
        """       
        i1, i2, i3 = map(self.get_atom_index, (a1, a2, a3))

        v1 = self.coords[i1] - self.coords[i2]
        v2 = self.coords[i3] - self.coords[i2]

        # dt = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # return np.arccos(dt)

        return _angle(v1, v2)

    def coord_subset(self, atoms: Iterable[AtomLike]) -> np.ndarray:
        """# `coord_subset`
        Returns the coordinates of a subset of atoms
        
        
        
        ## Parameters
        
        `atoms: Iterable[AtomLike]`
            The atoms of interest
        
        ## Returns
        
        `np.ndarray`
            The coordinates of the atoms
        
        ## Examples
            ``` Python
            mol.coord_subset([1, 2, 3]) = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            ```
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
        """# `dihedral`
        Compute the dihedral angle between four atoms in degrees
        
        
        
        ## Parameters
        
        `a1: AtomLike`
            First atom of interest
        `a2: AtomLike`
            Second atom of interest
        `a3: AtomLike`
            Third atom of interest
        `a4: AtomLike`
            Fourth atom of interest
        
        ## Returns
        
        `float`
            Dihedral angle between the four atoms in degrees
        
        ## Examples
            ``` Python
            mol.dihedral(1, 2, 3, 4) = 1.2
            ```
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
        """# `translate`
        Translate coordinates
        
        
        
        ## Parameters
        
        `vector: ArrayLike`
            Vector to translate the coordinates by

        ## Examples
            ``` Python
            mol.translate([1, 2, 3]) == mol.coords += [1, 2, 3] # True
            ```
        """        
        v = np.array(vector)
        self.coords += v[np.newaxis, :]

    def centroid(self) -> np.ndarray:
        """# `centroid`
        Centroid of the molecule
        
        
        
        ## Returns
        
        `np.ndarray`
            Molecule's Centroid
        
        ## Examples
            ``` Python
            mol = CartesianGeometry.from_smiles("C1CCCCC1")
            mol.centroid()
            array([0., 0., 0.])
            ```
        """        
        return np.average(self.coords, axis=0)
    

    def rmsd(self, other: CartesianGeometry, validate_elements=True):
        """# `rmsd`
        Compute the RMSD between two geometries
        
        
        
        ## Parameters
        
        `other: CartesianGeometry`
            Class instance of `CartesianGeometry`
        `validate_elements: bool`, optional, default: `True`
            Whether to validate the elements of the two molecules
        
        ## Raises
        
        `ValueError`
            Number of atoms in the two molecules are different
        `ValueError`
            Elements of the two molecules are different
        `NotImplementedError`
            RMSD calculation is not implemented
        """        
        if other.n_atoms != self.n_atoms:
            raise ValueError("Cannot compare geometries with different number of atoms")

        if validate_elements == True and self.elements != other.elements:
            raise ValueError(
                "Cannot compare two molecules with different lists of elements"
            )

        raise NotImplementedError

    def transform(self, _t_matrix: ArrayLike, /, validate=False):
        """# `transform`
        Transform the coordinates of the molecule
        
        
        
        ## Parameters
        
        `_t_matrix: ArrayLike`
            Matrix to transform the coordinates by
        `validate: bool`, optional, default: `False`
        
        ## Examples
            ``` Python
            mol.transform([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            ```
        """        
        t_matrix = np.array(_t_matrix)
        self.coords = t_matrix @ self.coords

    def del_atom(self, _a: AtomLike):
        """# `del_atom`
        Delete an atom from the molecule
        
        
        
        ## Parameters
        
        `_a: AtomLike`
            Atom to delete
        
        ## Examples
            ``` Python
            mol.del_atom(1) 
            ```
        """        
        ai = self.index_atom(_a)
        self._coords = np.delete(self._coords, ai, 0)
        super().del_atom(_a)