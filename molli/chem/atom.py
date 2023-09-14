# ================================================================================
# This file is part of
#      -----------
#      MOLLI 1.0.0
#      -----------
# (C) 2022 Alexander S. Shved and the Denmark laboratory
# University of Illinois at Urbana-Champaign, Department of Chemistry
# ================================================================================


"""
This file defines all constituent elements of 
"""
from __future__ import annotations
from typing import Any, List, Iterable, Generator, Callable
from enum import Enum, IntEnum
from dataclasses import dataclass, field, KW_ONLY
from collections import Counter, UserList
import numpy as np
from .. import data
from io import BytesIO
from functools import cache
from warnings import warn
import re
from bidict import bidict
import attrs


class Element(IntEnum):
    """
    Enumerates through elements of the periodic table
    """

    @classmethod
    @cache
    def get(cls, elt: ElementLike) -> Element:
        """
        Class method that provides more universial way of retrieving element instances.

        Args:
            elt (ElementLike): Desired element
            'int' will be interpreted as atomic number
            'str' will be interpreted as element name

        Returns:
            Element: Element instance

        Example Usage:
            >>> o = Element(8) # oxygen
            >>> f = Element["F"] # fluorine
            >>> f == Element.F # True
        """
        match elt:
            case Element() | int():
                return cls(elt)

            case str() as s:
                return cls[s.capitalize()]
            case _:
                return cls(elt)

    @property
    def symbol(self) -> str:
        """
        The symbol of the element

        Returns:
            str: A string representing the symbol of the element
        Example Usage:
            >>> my_element = Element(6) # carbon
            >>> print(my_element.symbol) # C
        """
        return self.name

    @property
    def z(self) -> int:
        """
        Atomic number of the element

        Returns:
            int: An interger representing the atomic number of the element

        Example Usage:
            >>> my_element = Element('C') # carbon
            >>> print(my_element.z) # 6
        """
        return self.value

    def get_property_value(self, property_name: str) -> int | str | float:
        """
        Retrieves desired value from dictionary key

        Args:
            property_name (str): Name of the property to be retrieved

        Returns:
            int | str | float: Value of the property

            *float* when asked for atomic weight, covalent radius, or Van der Waals radius

            *str* when asked for CPK coloring

            *int* when group value is requested

        Example Usage:
            >>> get_property_value("symbol") # C
        """
        prop_val = data.get("element", property_name, self.name, noexcept=True)

        return prop_val

    def __repr__(self) -> str:
        """
        Prints the name of the element

        Returns:
            str: Name of the element

        Example Usage:
            >>> my_element = Element("C") # carbon
            >>> my_element # carbon
        """
        return self.name

    @property
    def atomic_weight(self) -> float:
        """
        The atomic weight of the element

        Returns:
            float: A float representing the atomic weight of the element

        Example Usage:
            >>> my_element = Element("C") # carbon
            >>> print(my_element.atomic_weight) # 12.011
        """
        return self.get_property_value("atomic_weight")

    @property
    def cov_radius_1(self) -> float:
        """
        The covalent radius of a single bond

        Returns:
            float: A float representing the covalent radius of a single bond

        Example Usage:
            >>> my_element = Element("C") # carbon

        """
        return self.get_property_value("covalent_radius_1")

    @property
    def cov_radius_2(self) -> float:
        """
        The covalent radius of a double bond

        Returns:
            float: A float representing the covalent radius of a double bond
        """
        return self.get_property_value("covalent_radius_2")

    @property
    def cov_radius_3(self) -> float:
        """
        The covalent radius of a triple bond

        Returns:
            float: A float representing the covalent radius of a triple bond
        """
        return self.get_property_value("covalent_radius_3")

    @property
    def cov_radius_grimme(self) -> float:
        """
        This is the same definition of covalent radii; however, any metal element has been scaled down by 10% to allow for use
        with grimme's implementation of dftd-coordination number. (See DOI: 10.1063/1.3382344)

        Returns:
            float: A float representing the covalent radius of a single bond

        """
        return self.get_property_value("covalent_radius_grimme")

    @property
    def vdw_radius(self) -> float:
        """
        The Van der Waals radius

        Returns:
            float: A float representing the Van der Waals radius

        Example Usage:
            >>> my_element = Element("C") # carbon
            >>> print(my_element.vdw_radius) # 170 pm
        """
        return self.get_property_value("vdw_radius")

    @property
    def en_pauling(self) -> float:
        """
        The element's Pauling electronegativity

        Returns:
            float: A float representing the element's Pauling electronegativity

        Example Usage:
            >>> my_element = Element("C") # carbon
            >>> print(my_element.en_pauling) # 2.55
        """
        return self.get_property_value("en_pauling")

    @property
    def color_cpk(self) -> str:
        """
        The color of the element based on its classification according to the CPK color scheme

        Returns:
            str: A string representing the color of the element

        Example Usage:
            >>> my_element = Element("C") # carbon
            >>> print(my_element.color_cpk) # black
        """
        return self.get_property_value("color_cpk")

    @property
    def group(self) -> int:
        """element group identifier"""
        return self.get_property_value("group")

    def _serialize(self) -> int:
        """
        Serializes the element, allowing for it to be stored in a database more efficiently

        Returns:
            int: An integer representing the element
        """
        return self.value

    Unknown = 0

    # Regular elements
    H = 1
    He = 2
    Li = 3
    Be = 4
    B = 5
    C = 6
    N = 7
    O = 8
    F = 9
    Ne = 10
    Na = 11
    Mg = 12
    Al = 13
    Si = 14
    P = 15
    S = 16
    Cl = 17
    Ar = 18
    K = 19
    Ca = 20
    Sc = 21
    Ti = 22
    V = 23
    Cr = 24
    Mn = 25
    Fe = 26
    Co = 27
    Ni = 28
    Cu = 29
    Zn = 30
    Ga = 31
    Ge = 32
    As = 33
    Se = 34
    Br = 35
    Kr = 36
    Rb = 37
    Sr = 38
    Y = 39
    Zr = 40
    Nb = 41
    Mo = 42
    Tc = 43
    Ru = 44
    Rh = 45
    Pd = 46
    Ag = 47
    Cd = 48
    In = 49
    Sn = 50
    Sb = 51
    Te = 52
    I = 53
    Xe = 54
    Cs = 55
    Ba = 56
    La = 57
    Ce = 58
    Pr = 59
    Nd = 60
    Pm = 61
    Sm = 62
    Eu = 63
    Gd = 64
    Tb = 65
    Dy = 66
    Ho = 67
    Er = 68
    Tm = 69
    Yb = 70
    Lu = 71
    Hf = 72
    Ta = 73
    W = 74
    Re = 75
    Os = 76
    Ir = 77
    Pt = 78
    Au = 79
    Hg = 80
    Tl = 81
    Pb = 82
    Bi = 83
    Po = 84
    At = 85
    Rn = 86
    Fr = 87
    Ra = 88
    Ac = 89
    Th = 90
    Pa = 91
    U = 92
    Np = 93
    Pu = 94
    Am = 95
    Cm = 96
    Bk = 97
    Cf = 98
    Es = 99
    Fm = 100
    Md = 101
    No = 102
    Lr = 103
    Rf = 104
    Db = 105
    Sg = 106
    Bh = 107
    Hs = 108
    Mt = 109
    Ds = 110
    Rg = 111
    Cn = 112
    Nh = 113
    Fl = 114
    Mc = 115
    Lv = 116
    Ts = 117
    Og = 118


ElementLike = Element | str | int
"""
A type alias for anything that can be resolved as an element 

`str` is interpreted as element symbol 

`int` is interpreted as atomic number
"""
# class Atom:
#     """
#     Atom is a mutable object that is compared by id.
#     It stores atomic properties for the molecule.
#     Performance of the class was optimized through the use of __slots__
#     attrs allows for the storage of arbitrary atomic attributes
#     """

#     __slots__ = (
#         "_parent",
#         "element",
#         "label",
#         "isotope",
#         "traits",
#         "attrs",
#     )

#     def __init__(
#         self,
#         element: ElementLike = Element.Unknown,
#         isotope: int = None,
#         *,
#         traits: str = None,
#         label: str = None,
#         parent: Promolecule = None,
#         **attrs: Any,
#     ):

#         self.element = Element.get(element)
#         self.isotope = isotope
#         self.atype = atype
#         self.label = label
#         self.attrs = dict(**attrs)
#         self._parent = parent

#     def update(self, other: Atom):
#         self.element = other.element
#         self.label = other.label
#         self.atype = other.atype
#         self.isotope = other.isotope
#         self.dummy = other.dummy
#         self.attrs |= other.attrs

#     @property
#     def parent(self):
#         return self._parent


IMPLICIT_VALENCE = {
    1: 1,
    2: 2,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 3,
    14: 4,
    15: 3,
    16: 2,
    17: 1,
    18: 0,
}
"""This is the expected number of bonds for main group elements"""


class AtomType(IntEnum):
    """
    Enumerates through atom groups, hybridizations, and classifications.

    Example Usage:

        >>> ring = AtomType(2)  # Aromatic
        >>> ring == AtomType.Aromatic # True
    """

    Unknown = 0
    Regular = 1
    Aromatic = 2
    CoordinationCenter = 10
    Hypervalent = 20

    # Main Group Types
    MainGroup_sp3 = 31
    MainGroup_sp2 = 32
    MainGroup_sp = 33

    # Non-atom placeholders
    Dummy = 100
    AttachmentPoint = 101
    LonePair = 102

    # Wildcards

    # Specific atom classes
    C_Guanidinium = 201
    N_Amide = 202
    N_Nitro = 203
    N_Ammonium = 204
    O_Sulfoxide = 205
    O_Sulfone = 206
    O_Carboxylate = 207


class AtomStereo(IntEnum):
    """
    Enumerates through stereogenic categories

    Example Usage:
        >>> r = AtomStereo(10) # R
        >>> r == AtomStereo.R # True
    """

    Unknown = 0
    NotStereogenic = 1

    R = 10
    S = 11

    Delta = 20
    Lambda = 21


class AtomGeom(IntEnum):
    """
    Enumerates through atom geometries

    Example Usage:
        >>> a = ml.Atom("Si", isotope=29, geom=ml.AtomGeom.R4_Tetrahedral)
        >>> print(a.geom) # AtomGeom.R4_Tetrahedral

    """

    Unknown = 0
    R1 = 10

    R2 = 20
    R2_Linear = 21
    R2_Bent = 22
    # R2_Bent120 = 23
    # R2_Bent109 = 24
    # R2_Bent90 = 25

    R3 = 30
    R3_Planar = 31
    R3_Pyramidal = 32
    R3_TShape = 33

    R4 = 40
    R4_Tetrahedral = 41
    R4_SquarePlanar = 42
    R4_Seesaw = 43

    R5 = 50
    R5_TrigonalBipyramidal = 51
    R5_SquarePyramid = 52

    R6 = 60
    R6_Octahedral = 61


@attrs.define(slots=True, repr=True, hash=False, eq=False, weakref_slot=True)
class Atom:
    """
    Atom class is the most fundamental class that a molecule can have
    """

    element: Element = attrs.field(
        default=Element.Unknown,
        converter=Element.get,
        on_setattr=lambda s, a, v: Element.get(v),
    )

    isotope: int = attrs.field(default=None)

    label: str = attrs.field(
        default=None,
        # kw_only=True,
    )

    atype: AtomType = attrs.field(
        default=AtomType.Regular,
        # kw_only=True,
        repr=False
        # repr=lambda x: x.name,
    )
    stereo: AtomStereo = attrs.field(
        default=AtomStereo.Unknown,
        # kw_only=True,
        repr=False
        # repr=lambda x: x.name,
    )
    geom: AtomGeom = attrs.field(
        default=AtomGeom.Unknown,
        # kw_only=True,
        repr=False
        # repr=lambda x: x.name,
    )

    def evolve(self, **changes) -> Atom:
        """
        Evolves the atom into a new atom with the changes specified in the `changes` dictionary.

        Args:
            changes (dict): parameter changes to the Atom class

        Returns:
            Atom: a new Atom instance with the changes specified in the `changes` dictionary.

        Example Usage:
            >>> my_atom = ml.Atom(element = "C", atype = 1)
            >>> print(my_atom.atype) # Regular
            >>> my_atom = ml.Atom.evolve(atype = 2)
            >>> print(my_atom.atype) # Aromatic
        """
        return attrs.evolve(self, **changes)

    def as_dict(self) -> dict:
        """
        Returns the atom as a dictionary

        Returns:
            dict: a dictionary of the atom

        Example Usage:
            >>> my_atom = ml.Atom(element = 'H') # hydrogen
            >>> print(my_atom.as_dict()) # {'H': 1}
        """
        return attrs.asdict(self)

    def as_tuple(self):
        """
        Returns the atom as a tuple

        Returns:
            tuple: a tuple of the atom

        Example Usage:
            >>> my_atom = ml.Atom(element = "H") # hydrogen
            >>> print(my_atom.as_tuple()) # ("H", 1)
        """
        return attrs.astuple(self)

    @property
    def is_dummy(self) -> bool:
        """
        Checks if the atom type is a dummy

        Returns:
            bool: TRUE or FALSE

        Example Usage:
            >>> my_atom = ml.Atom.atype(100) # Dummy
            >>> print(my_atom.is_dummy) # TRUE
        """
        return self.atype == AtomType.Dummy

    @property
    def is_attachment_point(self) -> bool:
        """
        Checks if the atom is an attachment point

        Returns:
            bool: TRUE or FALSE
        
        """
        return self.atype == AtomType.AttachmentPoint

    @property
    def idx(self) -> int | None:
        """
        Returns the index of the current atom, if possible

        Returns:
            int | None: an integer representing the index of the atom
        """
        if self._parent is None:
            return None
        else:
            return self._parent.index_atom(self)

    # def __repr__(self):
    #     return f"Atom([{self.isotope or ''}{self.element!r}], label={self.label!r}, atype={self.atype!r})"

    def __eq__(self, other: AtomLike) -> bool:
        """
        Checks if two atoms are equal

        Args:
            other (AtomLike): an atom or an index of an atom

        Returns:
            bool: TRUE or FALSE

        Example Usage:
            >>> atom_1 = Atom(1) # hydrogen
            >>> atom_2 = Atom(5) # boron
            >>> atom_1 = atom_2 # FALSE
        """
        return self is other

    # This is a default version of hash function for objects
    def __hash__(self) -> int:
        return id(self)

    @property
    def implicit_valence(self) -> int:
        return IMPLICIT_VALENCE[self.element.group]

    @property
    def Z(self) -> int:
        """
        Atomic number of the element

        Returns:
            int: An interger representing the atomic number of the element

        Example Usage:
            >>> my_atom = ml.Element(C) # carbon
            >>> print(my_element.Z) # 6
        """
        return self.element.z

    @property
    def atomic_weight(self) -> float:
        return self.element.atomic_weight or 0.0

    @property
    def vdw_radius(self) -> float:
        """
        The Van der Waals radius

        Returns:
            float: A float representing the Van der Waals radius

        Example Usage:
            >>> my_atom = Element("H") # hydrogen
            >>> print(my_atom.vdw_radius) # 120 pm
        """
        return self.element.vdw_radius

    @property
    def cov_radius_1(self) -> float:
        """
        The covalent radius of a single bond

        Returns:
            float: A float representing the covalent radius of a single bond
        """
        return self.element.cov_radius_1

    @property
    def cov_radius_2(self) -> float:
        """
        The covalent radius of a double bond

        Returns:
            float: A float representing the covalent radius of a double bond

        """
        return self.element.cov_radius_2

    @property
    def cov_radius_3(self) -> float:
        """
        The covalent radius of a triple bond

        Returns:
            float: A float representing the covalent radius of a triple bond
        """
        return self.element.cov_radius_3

    @property
    def cov_radius_grimme(self) -> float:
        """
        This is the same definition of covalent radii; however, any metal element has been scaled down by 10% to allow for use
        with grimme's implementation of dftd-coordination number. (See DOI: 10.1063/1.3382344)

        Returns:
            float: A float representing the covalent radius of a single bond
        """
        return self.element.cov_radius_grimme

    @property
    def color_cpk(self) -> str:
        """
        The color of the element based on its classification according to the CPK color scheme

        Returns:
            str: A string representing the color of the element

        Example Usage:
            >>> my_atom = Element("C") # carbon
            >>> print(my_atom.color_cpk) # black
        """
        return self.element.color_cpk

    def set_mol2_type(self, m2t: str):
        if "." in m2t:
            mol2_elt, mol2_type = m2t.split(".", maxsplit=1)
        else:
            mol2_elt, mol2_type = m2t, None

        if mol2_elt != "Du":
            self.element = mol2_elt

        match mol2_type:
            case "4":
                if self.element != Element.N:
                    raise NotImplementedError(f"{mol2_type} not implemented for {mol2_elt}, only N")
                else:
                    self.atype = AtomType.N_Ammonium
                    self.geom = AtomGeom.R4_Tetrahedral

            case "3":
                self.atype = AtomType.MainGroup_sp3

            case "2":
                self.atype = AtomType.MainGroup_sp2

            case "1":
                self.atype = AtomType.MainGroup_sp

            case "ar":
                self.atype = AtomType.Aromatic

            case "am":
                if self.element == Element.N:
                    self.atype = AtomType.N_Amide
                    self.geom = AtomGeom.R3_Planar

            case "cat" if self.element == Element.C:
                self.atype = AtomType.C_Guanidinium
                self.geom = AtomGeom.R3_Planar

            case "pl3":
                self.geom = AtomGeom.R3_Planar

            case "co2" if self.element == Element.O:
                self.atype = AtomType.O_Carboxylate
                self.geom = AtomGeom.R1

            case "O" if self.element == Element.S:
                self.geom = AtomGeom.R3_Pyramidal
                self.atype = AtomType.O_Sulfoxide

            case "O2" if self.element == Element.S:
                self.geom = AtomGeom.R4_Tetrahedral
                self.atype = AtomType.O_Sulfone

            case "oh":
                self.geom = AtomGeom.R6_Octahedral

            case "th":
                self.geom = AtomGeom.R4_Tetrahedral

            case _ if mol2_elt == "Du":
                # This case if to handle Du.X
                self.element = (
                    Element[mol2_type] if mol2_type in Element._member_names_ else Element.Unknown
                )
                self.atype = AtomType.Dummy

            case _ if mol2_elt in Element._member_names_:
                pass

            case _:
                raise NotImplementedError(f"Cannot interpret mol2 type {m2t!r}")

    def get_mol2_type(self):
        match self.element, self.atype, self.geom:
            case _, AtomType.Regular, _:
                return f"{self.element.symbol}"

            case _, AtomType.Dummy, _:
                return f"Du.{self.element.symbol}"

            case _, AtomType.MainGroup_sp, _:
                return f"{self.element.symbol}.1"

            case _, AtomType.MainGroup_sp2, _:
                return f"{self.element.symbol}.2"

            case _, AtomType.MainGroup_sp3, _:
                return f"{self.element.symbol}.3"

            case Element.C, _, _:
                if self.atype == AtomType.Aromatic:
                    return f"{self.element.symbol}.ar"
                elif (self.atype == AtomType.C_Guanidinium) & (self.geom == AtomGeom.R3_Planar):
                    return f"{self.element.symbol}.cat"
                else:
                    return f"{self.element.symbol}"

            case Element.N, _, _:
                if (self.atype == AtomType.N_Ammonium) & (self.geom == AtomGeom.R4_Tetrahedral):
                    return f"{self.element.symbol}.4"
                elif (self.atype == AtomType.N_Amide) & (self.geom == AtomGeom.R3_Planar):
                    return f"{self.element.symbol}.am"
                elif self.atype == AtomType.Aromatic:
                    return f"{self.element.symbol}.ar"
                elif self.geom == AtomGeom.R3_Planar:
                    return f"{self.element.symbol}.pl3"
                else:
                    return f"{self.element.symbol}"

            case Element.O, _, _:
                if (self.atype == AtomType.O_Carboxylate) & (self.geom == AtomGeom.R1):
                    return f"{self.element.symbol}.co2"
                else:
                    return f"{self.element.symbol}"

            case Element.S, _, _:
                if (self.atype == AtomType.O_Sulfoxide) & (self.geom == AtomGeom.R3_Pyramidal):
                    return f"{self.element.symbol}.O"
                elif (self.atype == AtomType.O_Sulfone) & (self.geom == AtomGeom.R4_Tetrahedral):
                    return f"{self.element.symbol}.O2"
                else:
                    return f"{self.element.symbol}"

            case _, _, AtomGeom.R3_Planar:
                return f"{self.element.symbol}.pl3"

            case _, _, AtomGeom.R6_Octahedral:
                return f"{self.element.symbol}.oh"

            case _, _, AtomGeom.R4_Tetrahedral:
                return f"{self.element.symbol}.th"


AtomLike = Atom | int | str
"""
AtomLike can be an atom, its index, or a unique identifier
"""

"""
A type alias for anything that can be resolved as an Atom,

`int` is interpreted as an index of an atom
"""

RE_MOL_NAME = re.compile(r"[_a-zA-Z0-9]+")
RE_MOL_ILLEGAL = re.compile(r"[^_a-zA-Z0-9]")


class Promolecule:
    """
    This is a parent class that only employs methods that work on a *list of disconnected atoms with no structure or geometry assigned to them*.

    Any class that adds functionality on top of atom list should inherit this class
    for API compatibility reasons.
    """

    # __slots__ = ("_atoms", "_atom_index_cache", "_name", "charge", "mult")

    def __init__(
        self,
        other: Promolecule | Iterable[Atom] | Iterable[ElementLike] = None,
        /,
        *,
        n_atoms: int = 0,
        name: str = None,
        copy_atoms: bool = False,
        charge: int = None,
        mult: int = None,
        **kwds,  # mostly just for subclassing compatibility
    ):
        """
        Initialization of promolecule pre-allocates storage space.

        n_atoms is ignored in cas
        """
        self._atom_index_cache = bidict()

        self.name = name
        self.charge = charge or 0
        self.mult = mult or 1

        match other:
            case None:
                if n_atoms < 0:
                    raise ValueError("Cannot instantiate with negative number of atoms")

                self._atoms = list(Atom() for _ in range(n_atoms))
                self.name = name
                self.charge = charge or 0
                self.mult = mult or 1

            case Promolecule() as pm:
                self._atoms = list(a.evolve() for a in pm.atoms)
                if hasattr(pm, "name"):
                    self.name = name or pm.name
                if hasattr(pm, "charge"):
                    self.charge = charge or pm.charge
                if hasattr(pm, "mult"):
                    self.mult = mult or pm.mult

            case [*atoms] if all(isinstance(a, Atom) for a in atoms):
                if copy_atoms:
                    self._atoms = list(a.evolve() for a in atoms)
                else:
                    self._atoms = atoms

            case [*atoms] if all(isinstance(a, ElementLike) for a in atoms):
                self._atoms = list(Atom(a) for a in atoms)

            case _:
                raise NotImplementedError(f"Cannot interpret {other} of type {type(other)}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, formula={self.formula!r})"

    @property
    def attachment_points(self) -> List[Atom]:
        """
        List of atoms with attachment points

        Returns:
            List[Atom]: a list containing all atoms with attachment points

        Example Usage:
            >>> my_molecule = Promolecule("methylamine") # CH3NH2
            >>> print(my_molecule.attachment_points) # [Atom("N"), Atom("H"), Atom("H")]
        """
        return [a for a in self.atoms if a.atype == AtomType.AttachmentPoint]

    @property
    def n_attachment_points(self) -> int:
        """
        Total number of attachment points

        Returns:
            int: an integer representing the total number of attachment points

        Example Usage:
            >>> my_molecule = Promolecule("methylamine") # CH3NH2
            >>> print(my_molecule.n_attachment_points) # 3
        """
        return len(self.attachment_points)

    @property
    def name(self) -> str:
        """
        Promolecule name

        Returns:
            str: a string representing the name of the atom

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> print(my_molecule.name) # methylamine
        """
        return self._name

    @name.setter
    def name(self, value: str):
        if value is None or value is Ellipsis:
            self._name = "unknown"

        elif RE_MOL_NAME.fullmatch(value):
            self._name = value
        else:
            sub = RE_MOL_ILLEGAL.sub("_", value)
            self._name = sub
            warn(f"Replaced illegal characters in molecule name: {value} --> {sub}")

    @property
    def atoms(self) -> List[Atom]:
        """
        Atoms in the Promolecule

        Returns:
            List[Atom]: a list containing all Atom instances in the promolecule

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> print(my_molecule.atoms) # [Atom("C"), Atom("H"), Atom("H"), Atom("H"), Atom("N"), Atom("H"), Atom("H")]
        """
        return self._atoms

    @property
    def elements(self) -> List[Element]:
        """
        Elements in the promolecule

        Returns:
            List[Element]: a list containing all Element instances in the promolecule

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> print(my_molecule.elements) # [Element("C"), Element("H"), Element("H"), Element("H"), Element("N"), Element("H"), Element("H")]
        """
        return [a.element for a in self.atoms]

    @property
    def n_atoms(self) -> int:
        """
        Total number of atoms in the promolecule

        Returns:
            int: an integer representing the total number of atoms in the promolecule

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> print(my_molecule.n_atoms) # 7
        """
        return len(self.atoms)

    def get_atom(self, _a: AtomLike) -> Atom:
        """
        Fetches an atom from the promolecule

        Args:
            _a (AtomLike): an atom or an index of an atom

        Returns:
            Atom: an Atom instance

        Raises:
            ValueError: if the atom is not found
            ValueError: if the atom type is not found

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> print(my_molecule.get_atom(0)) # Atom("C")
        """
        match _a:
            case Atom():
                if _a in self.atoms:
                    return _a
                else:
                    raise ValueError(f"Atom {_a} does not belong to this molecule.")

            case int():
                return self._atoms[_a]

            case str():
                return next(self.yield_atoms_by_label(_a))

            case _:
                raise ValueError(f"Unable to fetch an atom with {type(_a)}: {_a}")

    def get_atoms(self, *_atoms: AtomLike) -> tuple[Atom]:
        """
        Fetches a list of atoms from the promolecule

        Args:
            _atoms (AtomLike): a list of atoms or indices of atoms

        Returns:
            tuple[Atom]: a tuple of Atom instances

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> print(my_molecule.get_atoms(0, 1, 2)) # (Atom("C"), Atom("H"), Atom("H"))
        """
        return tuple(map(self.get_atom, _atoms))

    def get_atom_index(self, _a: AtomLike) -> int:
        """
        Fetches the index of an atom in the promolecule

        Args:
            _a (AtomLike): an atom or an index of an atom

        Returns:
            int: an integer representing the index of the atom

        Raises:
            ValueError: if the atom is not found
            ValueError: if the atom type is not found

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> print(my_molecule.get_atom_index(Atom("C"))) # 0
        """
        match _a:
            case Atom():
                return self._atoms.index(_a)

            case int():
                if 0 <= _a < self.n_atoms:
                    return _a
                else:
                    raise ValueError(
                        f"Atom with index {_a} does not exist in a molecule"
                        f" with {self.n_atoms} atoms"
                    )

            case str():
                return self._atoms.index(next(self.yield_atoms_by_label(_a)))

            case _:
                raise ValueError(f"Unable to fetch an atom with {type(_a)}: {_a}")

    def get_atom_indices(self, *_atoms: AtomLike) -> tuple[int]:
        """
        Retrieves the indices of a list of atoms in the promolecule

        Args:
            _atoms (AtomLike): a list of atoms or indices of atoms

        Returns:
            tuple[int]: a tuple of integers representing the indices of the atoms

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> print(my_molecule.get_atom_indices(Atom("C"), Atom("H"), Atom("H"))) # (0, 1, 2)
        """
        return tuple(map(self.get_atom_index, _atoms))

    def del_atom(self, _a: AtomLike):
        """
        Deletes an atom from the promolecule

        Args:
            _a (AtomLike): an atom or an index of an atom

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> my_molecule = my_molecule.del_atom(0) # Atom("C")
            >>> print(my_molecule.get_atoms) # [Atom("H"), Atom("H"), Atom("H"), Atom("N"), Atom("H"), Atom("H")]
        """
        self._atoms.remove(_a)

    def append_atom(self, a: Atom):
        """
        Appends an atom to the promolecule

        Args:
            a (Atom): an Atom instance

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> my_molecule = my_molecule.append_atom(Atom("H"))
            >>> print(my_molecule.get_atoms) # [Atom("C"), Atom("H"), Atom("H"), Atom("H"), Atom("N"), Atom("H"), Atom("H"), Atom("H")]
        """
        self._atoms.append(a)

    def index_atom(self, _a: Atom) -> int:
        """
        Returns the index of an atom in the promolecule

        Args:
            _a (Atom): an Atom instance

        Returns:
            int: an integer representing the index of the atom

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> print(my_molecule.index_atom(Atom("C"))) # 0
        """
        return self._atoms.index(_a)

    # def yield_atom_indices(
    #     self, atoms: Iterable[AtomLike]
    # ) -> Generator[int, None, None]:
    #     for x in map(self.get_atom_index, atoms):
    #         yield x

    # def yield_atoms(
    #     self, atoms: Iterable[AtomLike]
    # ) -> Generator[Atom, None, None]:
    #     return map(self.get_atom, atoms)

    def yield_atoms_by_element(self, elt: Element | str | int) -> Generator[Atom, None, None]:
        for a in self.atoms:
            if a.element == Element.get(elt):
                yield a

    def yield_attachment_points(self) -> Generator[Atom, None, None]:
        """
        Yields atoms that contain attachment points

        Yield:
            Generator[Atom, None, None]: a generator of Atom instances

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> print(my_molecule.yield_attachment_points()) # [Atom("N"), Atom("H"), Atom("H")]
        """
        for a in self.atoms:
            if a.atype == AtomType.AttachmentPoint:
                yield a

    def get_attachment_points(self) -> tuple[Atom]:
        """
        Yields atoms that contain attachment points

        Yield:
            Generator[Atom, None, None]: a generator of Atom instances

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> print(my_molecule.yield_attachment_points()) # (Atom("N"), Atom("H"), Atom("H"))
        """
        return tuple(self.yield_attachment_points(self))

    def yield_atoms_by_label(self, lbl: str) -> Generator[Atom, None, None]:
        """
        Yields atoms that have a desired label

        Args:
            lbl (str): a string representing the label

        Yield:
            Generator[Atom, None, None]: a generator of Atom instances

        Example Usage:
            >>> my_molecule = Promolecule(CH3NH2)
            >>> print(my_molecule.yield_atoms_by_label(Regular)) # [Atom("C"), Atom("H"), Atom("H"), Atom("H"), Atom("N"), Atom("H"), Atom("H")]
        """
        for a in self.atoms:
            if a.label == lbl:
                yield a

    def sort_atoms(self, key: Callable[[Atom], int], reverse=False):
        raise NotImplementedError("TBD")

    # @n_atoms.setter
    # def n_atoms(self, other):
    #     raise SyntaxError("Cannot assign values to property <n_atoms>")

    @property
    def formula(self) -> str:
        """
        Molecular formula of promolecule

        Returns:
            str: a string representing the molecular formula

        Example Usage:
            >>> my_molecule = Promolecule("methylamine")
            >>> print(my_molecule.formula) # CH3NH2
        """
        if self.n_atoms > 0:
            ctr = Counter(x.element.symbol for x in self.atoms)
            f = []
            if "C" in ctr:
                f.append(f"""C{ctr.pop("C")}""")
            if "H" in ctr:
                f.append(f"""H{ctr.pop("H")}""")

            for x in sorted(ctr):
                f.append(f"""{x}{ctr.pop(x)}""")

            return " ".join(f)
        else:
            return "[no atoms]"

    @property
    def molecular_weight(self) -> float:
        """
        Molecular weight of the molecule

        **Warning**: currently there is no support for isotopic masses.

        Returns:
            float: a float representing the molecular weight

        Example Usage:
            >>> my_molecule = Promolecule("methylamine")
            >>> print(my_molecule.molecular_weight) # 31.057
        """
        return sum(a.atomic_weight for a in self.atoms)

    def label_atoms(self, template: str = "{e}{n0}"):
        """
        **Format code**:
        >>> `n0`: atom number (begin with 0)
        >>>  `n1`: atom number (begin with 1)
        >>> `e`: element symbol

        Args:
            template (str, optional): template for atom name. Defaults to `"{e}{n0}"`.
        """
        for i, a in enumerate(self.atoms):
            a.label = template.format(
                e=a.element.symbol,
                n0=i,
                n1=i + 1,
            )


PromoleculeLike = Promolecule | Iterable[Atom | ElementLike]
