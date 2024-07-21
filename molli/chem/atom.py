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
# `molli.chem.atom`

This submodule defines classes `Element`, `Atom`, `Promolecule`.
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
from weakref import ref


class Element(IntEnum):
    """The Element class is an Enumeration class used for calling elements
    in the periodic table

    Parameters
    ----------
    IntEnum :
        A parameter that accepts an integer enumeration from 0-118,
        with 0 being defined as an "Unknown" element
    """

    @classmethod
    def get(cls, elt: ElementLike) -> Element:
        """Class method that used to instantiate Molli Elements

        Parameters
        ----------
        elt : ElementLike | int | str
            - 'int' will be interpreted as atomic number (used as callable)
            - 'str' will be interpreted as element name (retrieved with indexing)

        Returns
        -------
        Element

        Examples
        -------
            >>> o = ml.Element(8) # Oxygen
            >>> f = ml.Element["F"] # Fluorine
            >>> f == ml.Element.F # True

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
        Returns
        -------
        str
            A string representing the symbol of the element

        Examples
        -------
            >>> ml.Element.C.symbol
            'C'
        """
        return self.name

    @property
    def z(self) -> int:
        """
        Returns
        -------
        int
            An integer representing the atomic number of the element

        Examples
        -------
            >>> ml.Element.C.z
            6
        """
        return self.value

    def get_property_value(self, property_name: str) -> int | str | float:
        """Retrieves desired property value from dictionary key

        Parameters
        ----------
        str
            Name of the property to be retrieved

        Returns
        -------
        property_value : int | str | float
            Value of the Property

        """
        prop_val = data.get("element", property_name, self.name, noexcept=True)

        return prop_val

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
           Name of the element

        Examples
        -------
            >>> ml.Element(1)
            H
        """
        return self.name

    @property
    def atomic_weight(self) -> float:
        """
        Returns
        -------
        float
            Atomic weight of the element
        Examples
        -------
            >>> ml.Element["C"].atomic_weight
            12.011
        """

        return self.get_property_value("atomic_weight")

    @property
    def cov_radius_1(self) -> float:
        """
        Returns
        -------
        float
            Represents the covalent radius of a single bond (based on DOI: 10.1021/jp5065819)
        Examples
        -------
            >>> ml.Element["Pb"].cov_radius_1
            1.44
        """
        return self.get_property_value("covalent_radius_1")

    @property
    def cov_radius_2(self) -> float:
        """Currently Not Implemented

        Returns
        -------
        float
            A float representing the covalent radius of a double bond
        """

        raise NotImplementedError(
            "Covalent Radius of Double Bonds Currently Not Implemented"
        )

        return self.get_property_value("covalent_radius_2")

    @property
    def cov_radius_3(self) -> float:
        """Currently Not Implemented

        Returns
        -------
        float
            A float representing the covalent radius of a triple bond
        """

        raise NotImplementedError(
            "Covalent Radius of Triple Bonds Currently Not Implemented"
        )

        return self.get_property_value("covalent_radius_3")

    @property
    def cov_radius_grimme(self) -> float:
        """This is the same definition of covalent radii; however, any metal element has been scaled down by 10% to allow for use
        with grimme's implementation of dftd-coordination number. (See DOI: 10.1063/1.3382344)

        Returns
        -------
        float
            Represents the covalent radius of a single bond by the Grimme Definition

        Examples
        -------
            >>> ml.Element["Pb"].cov_radius_grimme
            1.3
        """

        return self.get_property_value("covalent_radius_grimme")

    @property
    def vdw_radius(self) -> float:
        """
        Returns
        -------
        float
            The Bondi Van der Waals radius in Angstroms (based on DOIs: 10.1021/jp8111556 , 10.1021/j100785a001

        Examples
        -------
            >>> ml.Element["Pb"].vdw_radius
            2.02
        """

        return self.get_property_value("vdw_radius")

    @property
    def en_pauling(self) -> float:
        """Currently Not Implemented

        Returns
        -------
        float
            Represents the element's Pauling electronegativity

        """

        raise NotImplementedError("Pauling Electronegativity Currently Not Implemented")

        return self.get_property_value("en_pauling")

    @property
    def color_cpk(self) -> str:
        """
        Returns
        -------
        str
            Hex color code based on the CPK color scheme

        Examples
        -------
            >>> ml.Element["F"].color_cpk
            '#daa520'
        """

        return self.get_property_value("color_cpk")

    @property
    def group(self) -> int:
        """
        Returns
        -------
        int
            Group number from the periodic table

        Examples
        -------
            >>> ml.Element["He"].group
            18
        """

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

VALENCE_ELECTRONS = {
    13: 3,
    14: 4,
    15: 5,
    16: 6,
    17: 7,
    18: 8,
}


class AtomType(IntEnum):
    """The AtomType class is an Enumeration class for assigning atom
    types

    Parameters
    ----------
    IntEnum :
        Accepts integer enumerations for different atom types

    Examples
    -------
        >>> ml.AtomType(2) == ml.AtomType.Aromatic
        True
    """

    Unknown = 0
    Regular = 1
    Aromatic = 2
    CoordinationCenter = 10
    Hypervalent = 20

    sp3 = 31
    sp2 = 32
    sp = 33
    # These are discouraged, but mostly left for compatibility
    sp3d = 34
    sp3d2 = 35

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
    O_Nitro = 208


class AtomStereo(IntEnum):
    """The AtomStereo class is an Enumeration class used for stereogenic atom
    assignment

    Parameters
    ----------
    IntEnum :
        Accepts integer enumerations for different stereogenic assignments

    Examples
    -------
        >>> ml.AtomStereo(10) == ml.AtomStereo.R
        True
    """

    Unknown = 0
    NotStereogenic = 1

    R = 10
    S = 11

    Delta = 20
    Lambda = 21

    Tet_CW = 30
    Tet_CCW = 31


class AtomGeom(IntEnum):
    """The AtomGeom class is an Enumeration class for assigning atom geometries

    Parameters
    ----------
    IntEnum :
        Accepts integer enumerations for different atom geometries

    Examples
    -------
        >>> ml.AtomGeom(21) == ml.AtomGeom.R2_Linear
        True
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
    """The Atom class is the most fundamental class a molecule can have"""

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
        repr=False,
        # repr=lambda x: x.name,
    )
    stereo: AtomStereo = attrs.field(
        default=AtomStereo.Unknown,
        # kw_only=True,
        repr=False,
        # repr=lambda x: x.name,
    )
    geom: AtomGeom = attrs.field(
        default=AtomGeom.Unknown,
        # kw_only=True,
        repr=False,
        # repr=lambda x: x.name,
    )

    formal_charge: int = attrs.field(
        default=0,
    )

    # 2*Spin as an integer
    formal_spin: int = attrs.field(default=0)

    attrib: dict = attrs.field(factory=dict, repr=False)

    _parent = attrs.field(
        default=None,
        repr=False,
        converter=lambda x: x if x is None or isinstance(x, ref) else ref(x),
    )

    def __getstate__(self):
        # Serialization of objects should just exclude _parent and __weakref__
        return self.as_dict(schema=self.__slots__[:-2])

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    @property
    def parent(self):
        if self._parent is None:
            return None
        else:
            return self._parent()

    @parent.setter
    def parent(self, other):
        self._parent = other

    def evolve(self, **changes) -> Atom:
        """Evolves the atom into a new atom with the changes specified

            Returns
            -------
            Atom
                A new Atom instance with the changes specified

        Examples
        -------
            >>> my_atom = ml.Atom(element = 'C', atype = ml.AtomType.Regular)
            >>> new_atom = my_atom.evolve(atype = ml.AtomType.Aromatic)
            >>> new_atom.atype
            <AtomType.Aromatic: 2>
        """

        return attrs.evolve(self, **changes)

    def as_dict(self, schema: List[str] = None) -> dict:
        """Returns the atom as a dictionary

        Parameters
        ----------
        schema : List[str], optional
            Can be used to specify if only certain properties are desired, by default None

        Returns
        -------
        dict
            This dictionary contains properties of the associated atom

        Examples
        -------
            >>> ml.Atom(element='C').as_dict()
            {'element': C, 'isotope': None, ...}
            >>> ml.Atom(element='C').as_dict(['element','label','attrib'])
            {'element': C, 'label': None, 'attrib': {}}
        """
        if schema is None:
            return attrs.asdict(self)
        else:
            return {a: getattr(self, a, None) for a in schema}

    def as_tuple(self, schema: List[str] = None) -> tuple:
        """Returns the atom as a tuple

        Parameters
        ----------
        schema : List[str], optional
            Can be used to specify if only certain properties are desired, by default None

        Returns
        -------
        tuple
            This tuple contains properties of the associated atom

        Examples
        -------
            >>> ml.Atom(element='C').as_tuple()
            {C, None, ...}
            >>> ml.Atom(element='C').as_tuple(['element','label','attrib'])
            {C, None, {}}
        """

        if schema is None:
            return attrs.astuple(self)
        else:
            return tuple(getattr(self, a, None) for a in schema)

    @property
    def is_dummy(self) -> bool:
        """Checks if the atom type is Unknown or a Dummy

        Returns
        -------
        bool
            Returns True if a Dummy atom
        Examples
        -------
            >>> a = ml.Atom(element='Unknown', atype=ml.AtomType.Dummy)
            >>> a.is_dummy
            True
        """

        return self.atype == AtomType.Dummy

    @property
    def is_attachment_point(self) -> bool:
        """Checks if the atom is an attachment point

        Returns
        -------
        bool
            Returns True if an attachment point
        Examples
        -------
            >>> a = ml.Atom(element='Unknown', atype=ml.AtomType.AttachmentPoint)
            >>> a.is_attachment_point
            True
        """

        return self.atype == AtomType.AttachmentPoint

    @property
    def idx(self) -> int | None:
        """Returns the index of the atom if associated with a Molecule

        Returns
        -------
        int | None
            Represents the index of the atom

        Examples
        -------
        The index is undefined with no parent molecule
            >>> a = ml.Atom(element='C')
            >>> a.idx
            None

        The index is defined with dendrobine as the parent molecule
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.add_atom(a, coord=[0,0,0])
            >>> a.idx
            44 #
        """

        if self.parent is None:
            return None
        else:
            return self.parent.index_atom(self)

    # def __repr__(self):
    #     return f"Atom([{self.isotope or ''}{self.element!r}], label={self.label!r}, atype={self.atype!r})"

    def __eq__(self, other: AtomLike) -> bool:
        """Checks if two atoms are equal

        Parameters
        ----------
        other : AtomLike
            An atom or an index of an atom

        Returns
        -------
        bool
            Returns True if equal

        Examples
        -------
            >>> o = ml.Atom(element='O')
            >>> new_o = ml.Atom(element='O')
            >>> o == new_o
            False
        """

        return self is other

    # This is a default version of hash function for objects
    def __hash__(self) -> int:
        return id(self)

    @property
    def implicit_valence(self) -> int:
        """
        Returns
        -------
        int
            Integer based on the implicit valence

        Examples
        -------
            >>> ml.Atom(element='C').implicit_valence
            4

        """
        return IMPLICIT_VALENCE[self.element.group]

    @property
    def Z(self) -> int:
        """
        Returns
        -------
        int
            Returns an integer representing the atomic number of the element

        Examples
        -------
            >>> ml.Atom(element='C').Z
            6
        """

        return self.element.z

    @property
    def atomic_weight(self) -> float:
        """
        Returns
        -------
        float
            Atomic weight of the element
        Examples
        -------
            >>> ml.Atom(element='C').atomic_weight
            12.011
        """
        return self.element.atomic_weight or 0.0

    @property
    def vdw_radius(self) -> float:
        """
        Returns
        -------
        float
            The Bondi Van der Waals radius in Angstroms (based on DOIs: 10.1021/jp8111556 , 10.1021/j100785a001

        Examples
        -------
            >>> ml.Atom(element='Pb').vdw_radius
            2.02
        """
        return self.element.vdw_radius

    @property
    def cov_radius_1(self) -> float:
        """
        Returns
        -------
        float
            Represents the covalent radius of a single bond (based on DOI: 10.1021/jp5065819)
        Examples
        -------
            >>> ml.Element["Pb"].cov_radius_1
            1.44
        """
        return self.element.cov_radius_1

    @property
    def cov_radius_2(self) -> float:
        """Currently Not Implemented

        Returns
        -------
        float
            A float representing the covalent radius of a double bond
        """

        raise NotImplementedError(
            "Covalent Radius of Double Bonds Currently Not Implemented"
        )

        return self.element.cov_radius_2

    @property
    def cov_radius_3(self) -> float:
        """Currently Not Implemented

        Returns
        -------
        float
            A float representing the covalent radius of a triple bond
        """

        raise NotImplementedError(
            "Covalent Radius of Triple Bonds Currently Not Implemented"
        )
        return self.element.cov_radius_3

    @property
    def cov_radius_grimme(self) -> float:
        """This is the same definition of covalent radii; however, any metal element has been scaled down by 10% to allow for use
        with grimme's implementation of dftd-coordination number. (See DOI: 10.1063/1.3382344)

        Returns
        -------
        float
            Represents the covalent radius of a single bond by the Grimme Definition

        Examples
        -------
            >>> ml.Atom(element='Pb').cov_radius_grimme
            1.3
        """
        return self.element.cov_radius_grimme

    @property
    def color_cpk(self) -> str:
        """
        Returns
        -------
        str
            Hex color code based on the CPK color scheme

        Examples
        -------
            >>> ml.Atom(element='F').color_cpk
            '#daa520'
        """
        return self.element.color_cpk

    @property
    def valence_electrons(self) -> int:
        """Returns the number of valence electrons"""
        return VALENCE_ELECTRONS[self.element.group]

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
                self.atype = AtomType.sp3

            case "2":
                self.atype = AtomType.sp2

            case "1":
                self.atype = AtomType.sp

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

    def get_mol2_type(self) -> str:
        """Used to return the Sybyl Mol2 Type of an atom

        Returns
        -------
        str
            Returns the Sybyl Mol2 type of an atom

        Examples
        -------
            >>> unknown_molli_atom.get_mol2_type()
            >>> 'C.1' # Indicates it was ml.AtomType.MainGroup_sp
        """
        match self.element, self.atype, self.geom:
            case _, AtomType.Regular, _:
                return f"{self.element.symbol}"

            case _, AtomType.Dummy, _:
                return f"Du.{self.element.symbol}"

            case _, AtomType.sp, _:
                return f"{self.element.symbol}.1"

            case _, AtomType.sp2, _:
                return f"{self.element.symbol}.2"

            case _, AtomType.sp3, _:
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

            case _:
                return self.element.symbol


AtomLike = Atom | int | str | Element
"""
AtomLike can be an atom, its index, string, or element
"""


class Promolecule:
    """This is a parent class that only employs methods that work on a *list of
    disconnected atoms with no structure or geometry assigned to them*. Any class
    that adds functionality on top of atom list should inherit this class
    for API compatibility reasons.
    """

    __slots__ = (
        "_name",
        "_atoms",
        "_atomic_charges",
        "_bonds",
        "_adjacency",
        "_coords",
        "charge",
        "mult",
        "attrib",
        "_parent",
        "__weakref__",
    )

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
        attrib: dict = None,
        **kwds,  # mostly just for subclassing compatibility
    ):
        self.name = name
        self.charge = charge or 0
        self.mult = mult or 1
        self.attrib = attrib or dict()

        match other:
            case None:
                if n_atoms < 0:
                    raise ValueError("Cannot instantiate with negative number of atoms")

                self._atoms = list(Atom(parent=self) for _ in range(n_atoms))
                self.name = name
                self.charge = charge or 0
                self.mult = mult or 1

            case Promolecule() as pm:
                self._atoms = list(a.evolve(parent=self) for a in pm.atoms)
                if hasattr(pm, "name"):
                    self.name = name or pm.name
                if hasattr(pm, "charge"):
                    self.charge = charge or pm.charge
                if hasattr(pm, "mult"):
                    self.mult = mult or pm.mult
                self.attrib = pm.attrib.copy() | self.attrib

            case [*atoms] if all(isinstance(a, Atom) for a in atoms):
                if copy_atoms:
                    self._atoms = list(a.evolve(parent=self) for a in atoms)
                else:
                    self._atoms = atoms
                    for a in self._atoms:
                        a.parent = self

            case [*atoms] if all(isinstance(a, ElementLike) for a in atoms):
                self._atoms = list(Atom(a, parent=self) for a in atoms)

            case _:
                raise NotImplementedError(
                    f"Cannot interpret {other} of type {type(other)}"
                )

    def __getstate__(self):
        # Serialization of objects should just exclude _parent and __weakref__
        return {k: getattr(self, k, None) for k in self.__slots__[:-2]}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, formula={self.formula!r})"

    @property
    def attachment_points(self) -> List[Atom]:
        """
        Returns
        -------
        List[Atom]
            Returns a list of atoms whose AtomType is an AttachmentPoint

        Examples
        -------
        The Molecule class inherits attachment_points
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.attachment_points
            [] #There are no attachment points in the dendrobine file
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.attachment_points
            [] #There are no attachment points in the dendrobine file
        """

        return [a for a in self.atoms if a.atype == AtomType.AttachmentPoint]

    @property
    def n_attachment_points(self) -> int:
        """
        Returns
        -------
        int
            Returns the number of atoms whose AtomType is an AttachmentPoint

        Examples
        -------
        The Molecule class inherits n_attachment_points
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.n_attachment_points
            0 #There are no attachment points in the dendrobine file
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.n_attachment_points
            0 #There are no attachment points in the dendrobine file
        """

        return len(self.attachment_points)

    @property
    def name(self) -> str:
        """
        Returns
        -------
        str
            Returns the name of the Promolecule

        Examples
        -------
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.name
            dendrobine
        """

        return self._name

    @name.setter
    def name(self, value: str):
        if value is None or value is Ellipsis:
            self._name = "unknown"
        else:
            self._name = value

    @property
    def atoms(self) -> List[Atom]:
        """
        Returns
        -------
        List[Atom]
            Returns an ordered list of the atoms in the Promolecule instance

        Examples
        -------
        The Molecule class inherits atoms
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.atoms
            [Atom(element=N, ...),Atom(element=C, ...), ...]
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.atoms
            [Atom(element=N, ...),Atom(element=C, ...), ...]
        """

        return self._atoms

    @property
    def elements(self) -> List[Element]:
        """
        Returns
        -------
        List[Element]
            Returns an ordered list of the elements in the Promolecule instance
        Examples
        -------
        The Molecule class inherits elements
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.elements
            [N, C, C, C, C, ...]
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.elements
            [N, C, C, C, C, ...]
        """

        return [a.element for a in self.atoms]

    @property
    def n_atoms(self) -> int:
        """
        Returns
        -------
        int
            Returns the total number of atoms in the Promolecule instance

        Examples
        -------
        The Molecule class inherits n_atoms
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.n_atoms
            44
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.n_atoms
            44
        """

        return len(self.atoms)

    def get_atom(self, _a: AtomLike) -> Atom:
        """Fetches an atom from the Promolecule instance

        Parameters
        ----------
        _a : AtomLike
            An Atom, index, label, or Element. This will only return the first
            instance of the label or Element found.

        Returns
        -------
        Atom
            Returns the Atom instance

        Examples
        -------
        The Molecule class inherits get_atom()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.get_atom(10)
            Atom(element=O, isotope=None, label='O', formal_charge=0, formal_spin=0)
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.get_atom(10)
            Atom(element=O, isotope=None, label='O', formal_charge=0, formal_spin=0)
        """

        match _a:
            case Atom():
                if _a in self.atoms:
                    return _a
                else:
                    raise ValueError(f"Atom {_a} does not belong to this molecule.")

            case Element():
                return next(self.yield_atoms_by_element(_a))

            case int():
                return self._atoms[_a]

            case str():
                return next(self.yield_atoms_by_label(_a))

            case _:
                raise ValueError(f"Unable to fetch an atom with {type(_a)}: {_a}")

    def get_atoms(self, *_atoms: AtomLike) -> tuple[Atom]:
        """Fetches a tuple of Atoms from the Promolecule instance

        Returns
        -------
        tuple[Atom]
            Returns a Tuple of Atoms

        Examples
        -------
        The Molecule class inherits get_atoms()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.get_atoms(0,1,2)
            (Atom(element=N,...),Atom(element=C,...),Atom(element=C,...))
        Here is an example of getting atoms by element
            >>> dendrobine.get_atoms(*dendrobine.yield_atoms_by_element("H"))
            (Atom(element=H,...),Atom(element=H,...),Atom(element=H,...), ...)
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.get_atoms(0,1,2)
            (Atom(element=N,...),Atom(element=C,...),Atom(element=C,...))
        Here is an example of getting atoms by element
            >>> promol.get_atoms(*promol.yield_atoms_by_element("H"))
            (Atom(element=H,...),Atom(element=H,...),Atom(element=H,...), ...)
        """

        return tuple(map(self.get_atom, _atoms))

    def get_atom_index(self, _a: AtomLike) -> int:
        """Fetches the atom index from the promolecule

        Parameters
        ----------
        _a : AtomLike
            An atom, index, label, or Element. This will only return the first
            instance of the label or Element found.

        Returns
        -------
        int
            Returns the index of the atom

        Examples
        -------
        The Molecule class inherits get_atom_index()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.get_atom_index("N")
            0
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.get_atom_index("N")
            0
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
        """Fetches a tuple of indices from the Promolecule

        Returns
        -------
        tuple[int]
            Returns a tuple of indices

        Examples
        -------
        The Molecule class inherits get_atom_indices()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.get_atom_indices(*dendrobine.yield_atoms_by_element("H"))
            (16, 17, 18, 19, 23, 24, 25, ...)
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.get_atom_indices(*promol.yield_atoms_by_element("H"))
            (16, 17, 18, 19, 23, 24, 25, ...)
        """

        return tuple(map(self.get_atom_index, _atoms))

    def del_atom(self, _a: AtomLike) -> None:
        """Deletes an atom from the promolecule

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
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.del_atom(0)
            Atom(element=C, isotope=None, label='C', formal_charge=0, formal_spin=0)
        """
        a = self.get_atom(_a)
        self._atoms.remove(a)

    def append_atom(self, a: Atom) -> None:
        """Appends an atom to the Promolecule instance

        Parameters
        ----------
        a : Atom
            An atom instance to be added

        Examples
        -------
        The Molecule class inherits append_atom()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> new_atom = ml.Atom(element='H')
            >>> dendrobine.append_atom(new_atom)
            >>> dendrobine.get_atom(new_atom)
            Atom(element=H, isotope=None, label=None, formal_charge=0, formal_spin=0)
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.append_atom(new_atom)
            Atom(element=H, isotope=None, label=None, formal_charge=0, formal_spin=0)
        """

        self._atoms.append(a)
        a.parent = self

    def index_atom(self, _a: Atom) -> int:
        """Fetches the atom index from the Promolecule Instance

        Parameters
        ----------
        _a : Atom
            Must be an atom in the Promolecule instance list rather than AtomLike

        Returns
        -------
        int
            Returns the index of the atom

        Examples
        -------
        The Molecule class inherits index_atom()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> atom = dendrobine.get_atom("N")
            >>> dendrobine.index_atom(atom)
            0
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.index_atom(atom)
            0
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

    def yield_atoms_by_element(self, elt: ElementLike) -> Generator[Atom, None, None]:
        """Yields atoms based on their element

        Parameters
        ----------
        elt : ElementLike
            An element, integer, or float

        Yields
        ------
        Generator[Atom, None, None]
            Yields generator of Atom instances

        Examples
        -------
        The Molecule class inherits yield_atoms_by_element()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> generator = dendrobine.yield_atoms_by_element("H")
            <generator object Promolecule.yield_atoms_by_element at ...>
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> generator = promol.yield_atoms_by_element("H")
            <generator object Promolecule.yield_atoms_by_element at ...>
        """

        for a in self.atoms:
            if a.element == Element.get(elt):
                yield a

    def yield_attachment_points(self) -> Generator[Atom, None, None]:
        """Yields atoms that are attachment points

        Yields
        ------
        Generator[Atom, None, None]
            Yields generator of Atom instances that are attachment points

        Examples
        -------
        The Molecule class inherits yield_attachment_points()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> generator = dendrobine.yield_attachment_points()
            <generator object Promolecule.yield_attachment_points at ...>
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> generator = promol.yield_attachment_points()
            <generator object Promolecule.yield_attachment_points at ...>
        """

        for a in self.atoms:
            if a.atype == AtomType.AttachmentPoint:
                yield a

    def get_attachment_points(self) -> tuple[Atom]:
        """Gets tuple of atoms that are attachment points

        Returns
        -------
        tuple[Atom]
            Returns tuple of atoms that are attachment points

        Examples
        -------
        The Molecule class inherits get_attachment_points()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> generator = dendrobine.get_attachment_points()
            () #Dendrobine does not have attachment points
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> generator = promol.get_attachment_points()
            () #Dendrobine does not have attachment points
        """

        return tuple(self.yield_attachment_points())

    def yield_atoms_by_label(self, lbl: str) -> Generator[Atom, None, None]:
        """Yields atoms based on their labels

        Parameters
        ----------
        lbl : str
            A string representing a label

        Yields
        ------
        Generator[Atom, None, None]
            Yields a generator of Atom instances

        Examples
        -------
        The Molecule class inherits yield_atoms_by_element()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> generator = dendrobine.yield_atoms_by_label("H")
            <generator object Promolecule.yield_atoms_by_label at ...>
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> generator = promol.yield_atoms_by_label("H")
            <generator object Promolecule.yield_atoms_by_label at ...>
        """

        for a in self.atoms:
            if a.label == lbl:
                yield a

    def sort_atoms(self, key: Callable[[Atom], int], reverse=False):
        raise NotImplementedError("Sorting Atoms Currently Not Implemented")

    # @n_atoms.setter
    # def n_atoms(self, other):
    #     raise SyntaxError("Cannot assign values to property <n_atoms>")

    @property
    def formula(self) -> str:
        """
        Returns
        -------
        str
            A String representing the molecular formula of the Promolecule

        Examples
        -------
        The Molecule class inherits formula
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.formula
            C16 H25 N1 O2
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.formula
            C16 H25 N1 O2
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
        """Molecular weight of the molecule

        **Warning**: currently there is no support for isotopic masses.

        Returns
        -------
        float
            Returns float representing the molecular weight

        Examples
        -------
        The Molecule class inherits molecular_weight
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.molecular_weight
            263.381
        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.molecular_weight
            263.381
        """

        return sum(a.atomic_weight for a in self.atoms)

    def label_atoms(self, template: str = "{e}{n0}"):
        """Allows for unique labeling scheme of atoms in the Promolecule instance

        Parameters
        ----------
        template : str, optional
            String template for labeling scheme, by default "{e}{n0}":\n
            'e' = element, 'n0' = atom number (begin with 0),

        Examples
        -------
        The Molecule class inherits label_atoms()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.label_atoms('{e}{n1}')
            >>> dendrobine.atoms
            [Atom(...,label='N1'), Atom(...,label='C2'), Atom(...,label='C3')]

        If desired, one can work directly with Promolecule class instead
            >>> promol = ml.Promolecule(dendrobine)
            >>> promol.label_atoms('{e}{n1}')
            >>> promol.atoms
            [Atom(...,label='N1'), Atom(...,label='C2'), Atom(...,label='C3')]
        """
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
"""
PromoleculeLike can be a Promolecule, or an iterable of atoms or elements.
"""
