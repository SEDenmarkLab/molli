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
    # `Element`
    
    Enumerates through elements

    Args:
        IntEnum (cls): inherited class for enumeration of integers

    Potential Uses:
    ```Python
        o = Element(8) # oxygen
        f = Element["F"] # fluorine
        f == Element.F # True
    ```
    """

    @classmethod
    @cache
    def get(cls, elt: ElementLike) -> Element: 

        """
        # `get`
        Class method that provides more universial way of retrieving element instances.
        
        ## Parameters
        
        `elt: ElementLike`
            
            'int' will be interpreted as atomic number \n
            'str' will be interpreted as element name
        
        ## Returns
        
        `ElementLike`

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
        """# `symbol`
         The symbol of the element
        
        ## Returns
        
        `str` of the element's elemental symbol 
            
        """        
        return self.name

    @property
    def z(self) -> int:
        """# `z`
        Atomic number of the element
        
        ## Returns
        
        `int` of the element's atomic number
        """        
        "Atomic number"
        return self.value

    def get_property_value(self, property_name: str) -> int | str | float:

        """# `get_property_value`
        Retrieves desired value from dictionary key
        
        ## Parameters
        `property_name` value desired written as a `str`
        
        ## Returns
        
        `float` when asked for atomic weight, covalent radius, or Van der Waals radius \n
        `str` when asked for CPK coloring\n
        `int` when group value is requested
        """        
        prop_val = data.get("element", property_name, self.name, noexcept=True)

        return prop_val

    def __repr__(self) -> str:
        return self.name

    @property
    def atomic_weight(self) -> float:
        return self.get_property_value("atomic_weight")

    @property
    def cov_radius_1(self) -> float:
        return self.get_property_value("covalent_radius_1")

    @property
    def cov_radius_2(self) -> float:
        """Double bonded covalent radius"""
        return self.get_property_value("covalent_radius_2")

    @property
    def cov_radius_3(self) -> float:
        return self.get_property_value("covalent_radius_3")

    @property
    def cov_radius_grimme(self) -> float:
        return self.get_property_value("covalent_radius_grimme")

    @property
    def vdw_radius(self) -> float:
        return self.get_property_value("vdw_radius")

    @property
    def en_pauling(self) -> float:
        return self.get_property_value("en_pauling")

    @property
    def color_cpk(self) -> str:
        return self.get_property_value("color_cpk")

    @property
    def group(self) -> int | str:
        """element group identifier"""
        return self.get_property_value("group")

    def _serialize(self) -> int:
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
# `ElementLike`
A type alias for anything that can be resolved as an element \n
`str`is interpreted as element symbol \n
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


class AtomType(IntEnum):
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
    Unknown = 0
    NotStereogenic = 1

    R = 10
    S = 11

    Delta = 20
    Lambda = 21


class AtomGeom(IntEnum):
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

    def evolve(self, **changes):
        return attrs.evolve(self, **changes)

    def as_dict(self):
        return attrs.asdict(self)

    def as_tuple(self):
        return attrs.astuple(self)

    @property
    def is_dummy(self) -> bool:
        return self.atype == AtomType.Dummy

    @property
    def is_attachment_point(self) -> bool:
        return self.atype == AtomType.AttachmentPoint

    @property
    def idx(self) -> int | None:
        if self._parent is None:
            return None
        else:
            return self._parent.index_atom(self)

    # def __repr__(self):
    #     return f"Atom([{self.isotope or ''}{self.element!r}], label={self.label!r}, atype={self.atype!r})"

    def __eq__(self, other: AtomLike):
        return self is other

    # This is a default version of hash function for objects
    def __hash__(self) -> int:
        return id(self)

    @property
    def Z(self) -> int:
        """Returns the atomic number (nuclear charge) of the element"""
        return self.element.z

    @property
    def vdw_radius(self) -> float:
        return self.element.vdw_radius

    @property
    def cov_radius_1(self) -> float:
        return self.element.cov_radius_1

    @property
    def cov_radius_2(self) -> float:
        return self.element.cov_radius_2

    @property
    def cov_radius_3(self) -> float:
        return self.element.cov_radius_3

    @property
    def cov_radius_grimme(self) -> float:
        """
        This is the same definition of covalent radii; however, any metal element has been scaled down by 10% to allow for use
        with grimme's implementation of dftd-coordination number. (See DOI: 10.1063/1.3382344)
        """
        return self.element.cov_radius_grimme

    @property
    def color_cpk(self) -> str:
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
                    raise NotImplementedError(
                        f"{mol2_type} not implemented for {mol2_elt}, only N"
                    )
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
                    Element[mol2_type]
                    if mol2_type in Element._member_names_
                    else Element.Unknown
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
                elif (self.atype == AtomType.C_Guanidinium) & (
                    self.geom == AtomGeom.R3_Planar
                ):
                    return f"{self.element.symbol}.cat"
                else:
                    return f"{self.element.symbol}"

            case Element.N, _, _:
                if (self.atype == AtomType.N_Ammonium) & (
                    self.geom == AtomGeom.R4_Tetrahedral
                ):
                    return f"{self.element.symbol}.4"
                elif (self.atype == AtomType.N_Amide) & (
                    self.geom == AtomGeom.R3_Planar
                ):
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
                if (self.atype == AtomType.O_Sulfoxide) & (
                    self.geom == AtomGeom.R3_Pyramidal
                ):
                    return f"{self.element.symbol}.O"
                elif (self.atype == AtomType.O_Sulfone) & (
                    self.geom == AtomGeom.R4_Tetrahedral
                ):
                    return f"{self.element.symbol}.O2"
                else:
                    return f"{self.element.symbol}"

            case _, _, AtomGeom.R3_Planar:
                return f"{self.element.symbol}.pl3"

            case _, _, AtomGeom.R6_Octahedral:
                return f"{self.element.symbol}.oh"

            case _, _, AtomGeom.R4_Tetrahedral:
                return f"{self.element.symbol}.th"


AtomLike = Atom | int

RE_MOL_NAME = re.compile(r"[_a-zA-Z0-9]+")
RE_MOL_ILLEGAL = re.compile(r"[^_a-zA-Z0-9]")


class Promolecule:
    """
    # `Promolecule` 
    This is a parent class that only employs methods that work on a \n
    **list of disconnected atoms with no structure or geometry assigned to them.** \n
    Any class that adds functionality on top of atom list should inherit this class \n
    for API compatibility reasons.
    """

    __slots__ = ("_atoms", "_atom_index_cache", "_name", "charge", "mult")

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
                self.name = name or pm.name
                self.charge = charge or pm.charge
                self.mult = mult or pm.mult

            case [*atoms] if all(isinstance(a, Atom) for a in atoms):
                if copy_atoms:
                    self._atoms = list(a.evolve() for a in atoms)
                else:
                    self._atoms = atoms

            case [*atoms] if all(isinstance(a, ElementLike) for a in atoms):
                self._atoms = list(Atom(a) for a in atoms)

            case _:
                raise NotImplementedError(
                    f"Cannot interpret {other} of type {type(other)}"
                )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}," f" formula={self.formula!r})"
        )

    @property
    def attachment_points(self) -> List[Atom]:
        return [a for a in self.atoms if a.atype == AtomType.AttachmentPoint]

    @property
    def n_attachment_points(self) -> int:
        return len(self.attachment_points)

    @property
    def name(self) -> str:
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
            warn(f"Replaced illegal characters in molecule name: {value} -->" f" {sub}")

    @property
    def atoms(self) -> List[Atom]:
        """List of atoms in the promolecule"""
        return self._atoms

    @property
    def elements(self) -> List[Element]:
        """List of elements in the protomolecule"""
        return [a.element for a in self.atoms]

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the promolecule"""
        return len(self.atoms)

    def get_atom(self, _a: AtomLike) -> Atom:
        match _a:
            case Atom():
                if _a in self.atoms:
                    return _a
                else:
                    raise ValueError(f"Atom {_a} does not belong to this molecule.")

            case int():
                return self._atoms[_a]

            case _:
                raise ValueError(f"Unable to fetch an atom with {type(_a)}: {_a}")

    def get_atoms(self, *_atoms: AtomLike) -> tuple[Atom]:
        return tuple(map(self.get_atom, _atoms))

    def get_atom_index(self, _a: AtomLike):
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

            case _:
                raise ValueError(f"Unable to fetch an atom with {type(_a)}: {_a}")

    def get_atom_indices(self, *_atoms: AtomLike) -> tuple[int]:
        return tuple(map(self.get_atom_index, _atoms))

    def del_atom(self, _a: AtomLike):
        self._atoms.remove(_a)

    def append_atom(self, a: Atom):
        self._atoms.append(a)

    def index_atom(self, _a: Atom) -> int:
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

    def yield_atoms_by_element(
        self, elt: Element | str | int
    ) -> Generator[Atom, None, None]:
        for a in self.atoms:
            if a.element == Element.get(elt):
                yield a

    def yield_attachment_points(self):
        for a in self.atoms:
            if a.atype == AtomType.AttachmentPoint:
                yield a

    def get_attachment_points(self):
        return tuple(self.yield_attachment_points(self))

    def yield_atoms_by_label(self, lbl: str) -> Generator[Atom, None, None]:
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
        # `molecular_weight`
        Molecular weight of the molecule

        Warning: currently there is no support for isotopic masses.

        ## Returns

        `float`
            molecular weight in Da
        """
        _mw = 0.0
        for a in self.atoms:
            _mw += a.element.atomic_weight
        return _mw

    def label_atoms(self, template: str = "{e}{n0}"):
        """# `label_atoms`

        Format code:
        - `n0`: atom number (begin with 0)
        - `n1`: atom number (begin with 1)
        - `e`: element symbol

        ## Parameters

        `template: str`, optional, default: `"{e}{n}"`
            Template for atom name
        """
        for i, a in enumerate(self.atoms):
            a.label = template.format(
                e=a.element.symbol,
                n0=i,
                n1=i + 1,
            )


PromoleculeLike = Promolecule | Iterable[Atom | ElementLike]
