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
from enum import Enum
from dataclasses import dataclass, field, KW_ONLY
from collections import Counter, UserList
import numpy as np
from .. import data
from io import BytesIO
from functools import cache
from warnings import warn
import re


class Element(Enum):
    """Element enumerator"""

    @classmethod
    @cache
    def get(cls, elt: ElementLike):
        """More universal way of retrieving element instances"""
        match elt:
            case Element() | int():
                return cls(elt)
            case str():
                return cls[elt]
            case bytes():
                return cls[elt.decode("ascii")]
            case _:
                return cls(elt)

    @property
    def symbol(self):
        "Element symbol"
        return self.name

    @property
    def z(self):
        "Atomic number"
        return self.value

    def get_property_value(self, property_name: str):
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
    def vdw_radius(self) -> float:
        return self.get_property_value("vdw_radius")

    @property
    def en_pauling(self) -> float:
        return self.get_property_value("en_pauling")

    @property
    def color_cpk(self) -> str:
        return self.get_property_value("color_cpk")

    def _serialize(self) -> int:
        return self.value

    # Just for compatibility reasons
    LonePair = -1
    Lp = LonePair
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
This is a type alias for anything that can be resolved as an element
String is interpreted as element symbol
Integer is interpreted as atomic number
"""


class Atom:
    """
    Atom is a mutable object that is compared by id.
    It stores atomic properties for the molecule.
    Performance of the class was optimized through the use of __slots__
    """

    __slots__ = (
        "element",
        "label",
        "isotope",
        "stereo",
        "dummy",
    )

    def __init__(
        self,
        element: ElementLike = Element.Unknown,
        label: str = ...,
        *,
        isotope: int = -1,
        dummy: bool = False,
        stereo: bool = False,
    ):
        self.element = Element.get(element)
        self.isotope = isotope
        self.dummy = dummy
        self.stereo = stereo
        self.label = label if label is not Ellipsis else self.element.symbol

    @classmethod
    def add_to(
        cls: type[Atom],
        parent: Promolecule,
        /,
        element: ElementLike = Element.Unknown,
        label: str = ...,
        isotope: int = -1,
        dummy: bool = False,
        stereo: bool = False,
    ):
        a = cls(
            element=element,
            label=label,
            isotope=isotope,
            dummy=dummy,
            stereo=stereo,
        )

        parent.append_atom(a)
        return a

    def __repr__(self):

        _i = self.isotope if self.isotope > 0 else ""
        _d = " #" if self.dummy else ""
        _s = " *" if self.stereo else ""

        inner = f"{_i}{self.element.symbol}"

        inner += f""", label='{self.label}'{_d}{_s}"""

        return f"Atom({inner})"

    # def __eq__(self, other: AtomLike):
    #     return self is other

    # This is a default version of hash function for objects
    # def __hash__(self) -> int:
    #     return id(self) >> 4  # Equivalent to id(self) // 16

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
    def color_cpk(self) -> str:
        return self.element.color_cpk


AtomLike = Atom | int

RE_MOL_NAME = re.compile(r"[_a-zA-Z0-9]+")
RE_MOL_ILLEGAL = re.compile(r"[^_a-zA-Z0-9]")


class Promolecule:
    """
    This is a parent class that only employs methods that work on a
    **list of disconnected atoms with no structure or geometry assigned to them.**

    Any class that adds functionality on top of atom list should inherit this class
    for API compatibility reasons.
    """

    __slots__ = ("_atoms", "_name")

    def __init__(self, n_atoms: int = 0, name="unnamed"):
        """
        Initialization of promolecule pre-allocates storage space
        """
        if n_atoms < 0:
            raise ValueError("Cannot instantiate with negative number of atoms")

        self._atoms = list()
        for _ in range(n_atoms):
            Atom.add_to(self)

        self.name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str | bytes):
        if isinstance(value, bytes):
            _value = value.decode("ascii")
        else:
            _value = value

        if RE_MOL_NAME.fullmatch(_value):
            self._name = _value
        else:
            sub = RE_MOL_ILLEGAL.sub("_", _value)
            self._name = sub
            warn(f"Replaced illegal characters in molecule name: {_value} --> {sub}")

            # raise ValueError(
            #     f"Inappropriate name for a molecule: {value}. Must match {RE_MOL_NAME.pattern}"
            # )

    @property
    def atoms(self) -> List[Atom]:
        """List of atoms in the promolecule"""
        return self._atoms

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the promolecule"""
        return len(self.atoms)

    def append_atom(self, a: Atom):
        self._atoms.append(a)

    def del_atom(self, a: AtomLike):
        match a:
            case Atom():
                idx = self.index_atom(a)
            case int():
                idx = a

        del self._atoms[idx]

    def get_atom(self, a: AtomLike) -> Atom:
        match a:
            case Atom():
                if a in self.atoms:
                    return a
                else:
                    raise ValueError(f"Atom {a} does not belong to this molecule.")

            case int():
                return self._atoms[a]

            case _:
                raise ValueError(f"Unable to fetch an atom with {type(a)}: {a}")

    def index_atom(self, a: AtomLike) -> int:
        # return self._atoms.index(a)
        match a:
            case Atom():
                return self._atoms.index(a)

            case int():
                if 0 <= a < self.n_atoms:
                    return a
                else:
                    raise ValueError(
                        f"Atom with index {a} does not exist in a molecule with {self.n_atoms} atoms"
                    )

            case _:
                raise ValueError(f"Unable to fetch an atom with {type(a)}: {a}")

    def yield_atom_indices(
        self, atoms: Iterable[AtomLike]
    ) -> Generator[int, None, None]:
        for a in atoms:
            yield self.index_atom(a)

    def yield_atoms(self, atoms: Iterable[AtomLike]) -> Generator[Atom, None, None]:
        for a in atoms:
            yield self.get_atom(a)

    def get_atoms_by_element(
        self, elt: Element | str | int
    ) -> Generator[Atom, None, None]:
        for a in self.atoms:
            if a.element == Element.get(elt):
                yield a

    def get_atoms_by_label(self, lbl: str) -> Generator[Atom, None, None]:
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
        ctr = Counter(x.element.symbol for x in self.atoms)
        f = []
        if "C" in ctr:
            f.append(f"""C{ctr.pop("C")}""")
        if "H" in ctr:
            f.append(f"""H{ctr.pop("H")}""")

        for x in sorted(ctr):
            f.append(f"""{x}{ctr.pop(x)}""")

        return " ".join(f)

    @property
    def molecular_weight(self) -> float:
        _mw = 0.0
        for a in self.atoms:
            _mw += a.element.atomic_weight
        return _mw


PromoleculeLike = Promolecule | Iterable[Atom | ElementLike]
