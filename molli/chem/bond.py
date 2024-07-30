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
# `molli.chem.bond`

This submodule defines classes `Bond`, `Connectivity` and others.
"""

from __future__ import annotations
from . import (
    Atom,
    AtomType,
    AtomStereo,
    Element,
    AtomLike,
    Promolecule,
    PromoleculeLike,
)
from dataclasses import dataclass, field, KW_ONLY
from typing import Iterable, List, Generator, Tuple, Any, Callable
from copy import deepcopy
from enum import IntEnum
from collections import deque
from struct import pack, unpack, Struct
from io import BytesIO
import attrs
from bidict import bidict
from functools import cache
from weakref import ref, WeakKeyDictionary, WeakSet
import networkx as nx


class BondType(IntEnum):
    """The BondType class is an Enumeration class for assigning bond types

    Parameters
    ----------
    IntEnum :
        Accepts integer enumerations for different bond types

    Examples
    -------
        >>> ml.BondType(20) == ml.BondType.Aromatic
        True
    """

    Unknown = 0
    Single = 1
    Double = 2
    Triple = 3
    Quadruple = 4
    Quintuple = 5
    Sextuple = 6

    Dummy = 10
    NotConnected = 11

    Aromatic = 20
    Amide = 21

    Ligand = 98
    FractionalOrder = 99

    H_Donor = 100
    H_Acceptor = 101


class BondStereo(IntEnum):
    """The BondStereo class is an Enumeration class for assigning bond geometry

    Parameters
    ----------
    IntEnum :
        Accepts integer enumerations for different bond geometry

    Examples
    -------
        >>> ml.BondStereo(10) == ml.BondStereo.E
        True
    """

    Unknown = 0
    NotStereogenic = 1

    E = 10
    Z = 11

    Trans = E
    Cis = Z

    Axial_R = 20
    Axial_S = 21

    R = Axial_R
    S = Axial_S


# orders of 4, 5, 6 are not canonical per the mol2 definition file
MOL2_BOND_TYPE_MAP = bidict(
    {
        "1": BondType.Single,
        "2": BondType.Double,
        "3": BondType.Triple,
        "4": BondType.Quadruple,
        "5": BondType.Quintuple,
        "6": BondType.Sextuple,
        "ar": BondType.Aromatic,
        "am": BondType.Amide,
        "du": BondType.Dummy,
        "un": BondType.Unknown,
        "nc": BondType.NotConnected,
    }
)


@attrs.define(slots=True, repr=True, hash=False, eq=False, weakref_slot=True)
class Bond:
    """The class for bonds in the MOLLI package. a1 and a2 are the atoms that the
    bond connects and are interchangeable. The atoms are ordered upon initialization.
    """

    a1: Atom = attrs.field(repr=lambda a: a.idx or a)
    a2: Atom = attrs.field(repr=lambda a: a.idx or a)

    label: str = attrs.field(
        default=None,
    )

    btype: BondType = attrs.field(
        default=BondType.Single,
        repr=lambda x: x.name if hasattr(x, 'name') else x,
    )

    stereo: BondStereo = attrs.field(
        default=BondStereo.Unknown,
        repr=lambda x: x.name if hasattr(x, 'name') else x,
    )

    f_order: float = attrs.field(
        default=1.0,
        converter=float,
    )

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

    def evolve(self, **changes) -> Bond:
        """Evolves the bond into a new bond with the changes specified

        Returns
        -------
        Bond
            A new Bond instance with the changes specified

        Examples
        -------
            >>> benzene = ml.Molecule.load_mol2(ml.files.benzene_mol2)
            >>> bond = benzene.get_bond(0)
            >>> bond.btype
            <BondType.Aromatic: 20>
            >>> new_bond = bond.evolve(btype = ml.BondType.Double)
            >>> new_bond.btype
            <BondType.Double: 2>
        """

        return attrs.evolve(self, **changes)

    @property
    def order(self) -> float:
        """
        Returns
        -------
        float
            Returns the bond order as a float

        Examples
        -------
        Bonds default to 1.0 when not specified
            >>> bond = ml.Bond(a1 = ml.Atom("C"), a2= ml.Atom("C"))
            >>> bond.order
            1.0
        Aromatic Bonds default to 1.5
            >>> benzene = ml.Molecule.load_mol2(ml.files.benzene_mol2)
            >>> bond = benzene.get_bond(0)
            >>> bond.order
            1.5
        """

        # if self.btype == BondType.FractionalOrder:
        #     return self._order
        # elif 0 < self.btype < 9:
        #     return float(self.btype)
        # elif self.btype in {BondType.Aromatic}

        match self.btype:
            case 0 | 1 | 2 | 3 | 4 | 5 | 6 as b:
                return float(b)

            case BondType.Aromatic:
                return 1.5

            case BondType.FractionalOrder:
                return self.f_order

            case BondType.H_Acceptor:
                return 0.0

            case BondType.Dummy | BondType.Ligand | BondType.NotConnected:
                return 0.0

            case _:
                return 1.0

    def as_dict(self, schema: List[str] = None) -> dict:
        """Returns the bond as a dictionary

        Parameters
        ----------
        schema : List[str], optional
            Can be used to specify if only certain properties are desired,
            by default None

        Returns
        -------
        dict
            This dictionary contains properties of the associated bond

        Examples
        -------
            >>> ml.Bond(a1 = ml.Atom("C"), a2= ml.Atom("C")).as_dict()
            {'a1':{'element': C...}, 'a2': {'element': C ...}, 'label': None, ...}
            >>> bond = ml.Bond(a1 = ml.Atom("C"), a2= ml.Atom("C")).as_dict(
                ['a1','stereo','attrib']
                )
            {'a1':{'element': C...}, 'stereo': <BondStereo.Unknown: 0>,
            'attrib': {}}}
        """

        if schema is None:
            return attrs.asdict(self)
        else:
            return {a: getattr(self, a, None) for a in schema}

    def as_tuple(self, schema: List[str] = None) -> tuple:
        """Returns the bond as a tuple

        Parameters
        ----------
        schema : List[str], optional
            Can be used to specify if only certain properties are desired,
            by default None

        Returns
        -------
        tuple
            This tuple contains properties of the associated bond,
            also returning a1 and a2 as tuples

        Examples
        -------
            >>> ml.Bond(a1 = ml.Atom("C"), a2= ml.Atom("C")).as_tuple()
            ((C,...),(C,...), None, <BondType.Single: 1>, ...)
            >>> bond = ml.Bond(a1 = ml.Atom("C"), a2= ml.Atom("C")).as_tuple(
                ['a1','stereo','attrib']
                )
            ((C,...),<BondStereo.Unknown: 0>, {})
        """

        if schema is None:
            return attrs.astuple(self)
        else:
            return tuple(getattr(self, a, None) for a in schema)

    def __contains__(self, other: Atom) -> bool:
        """Checks if atom is in the bond

        Parameters
        ----------
        other : Atom
            Atom to check

        Returns
        -------
        bool
            Returns True if atom is in the bond

        Examples
        -------
            >>> bond = ml.Bond(a1 = ml.Atom("C"), a2= ml.Atom("C"))
            >>> new_atom = ml.Atom("C")
            >>> new_atom in bond
            False
        """

        return other in {self.a1, self.a2}

    def __eq__(self, other: Bond | set) -> bool:
        """Checks if two bonds are equal

        Parameters
        ----------
        other : Bond | set
            Bond or set to check

        Returns
        -------
        bool
            Returns True if bond is the same

        Examples
        -------
            >>> a1, a2 = ml.Atom("C"), ml.Atom("C")
            >>> bond1 = ml.Bond(a1,a2)
            >>> bond2 = ml.Bond(a2,a1)
            >>> bond1 == bond2
            True
        """

        # return self is other
        match other:
            case Bond():
                return {self.a1, self.a2} == {other.a1, other.a2}
            case list() | set() as l:
                a1, a2 = l
                return {self.a1, self.a2} == {a1, a2}
            case _:
                raise ValueError(f"Cannot equate <{type(other)}: {other}>, {self}")

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            Prints bond as string

        Examples
        -------
            >>> bond = ml.Bond(a1 = ml.Atom("C"), a2= ml.Atom("C"))
            >>> bond
            Bond(a1=Atom(element=C, ...), a2=Atom(element=C, ...), label=None, ...)
        """

        return f"Bond({self.a1}, {self.a2}, order={self.order})"

    # This mimics the default behavior of object instances in python
    def __hash__(self) -> int:
        return id(self) >> 4

    def __mod__(self, a: Atom) -> Atom:
        """Allows % to be used on a bond in combination with an atom to
        return the other atom of the bond.

        Parameters
        ----------
        a : Atom
            One atom of the bond

        Returns
        -------
        Atom
            The other Atom instance of the bond

        Examples
        -------
            >>> a1, a2 = ml.Atom("C", label='a1'), ml.Atom("H", label="a2")
            >>> bond = ml.Bond(a1, a2)
            >>> bond % a1
            Atom(element=H, ... label='a2', ...)
        """

        if self.a1 == a:
            return self.a2
        elif self.a2 == a:
            return self.a1
        else:
            raise ValueError("Atom is not a part of this bond")

    @property
    def expected_length(self) -> float:
        """
        Returns
        -------
        float
            Returns an estimated value of a bond connection between two atoms
            based on the covalent radius of a single bond

        Examples
        -------
            >>> a1, a2 = ml.Atom("Pb", label='a1'), ml.Atom("H", label="a2")
            >>> bond = ml.Bond(a1, a2)
            >>> bond.expected_length
            1.76 # (Pb = 1.44, H = 0.32)

        """
        r1 = self.a1.cov_radius_1
        r2 = self.a2.cov_radius_1
        return (r1 or Element.C.cov_radius_1) + (r2 or Element.C.cov_radius_1)

    @cache
    def set_mol2_type(self, m2t: str):
        self.btype = MOL2_BOND_TYPE_MAP[m2t]

    def get_mol2_type(self) -> str:
        """Used to return the Sybyl Mol2 Type of a bond

        Returns
        -------
        str
            Returns the Sybyl Mol2 type of a bond

        Examples
        -------
            >>> unknown_molli_bond.get_mol2_type()
            'am' # Indicates it was an amide bond
        """
        match self.btype:
            case BondType.Single:
                # print()
                return MOL2_BOND_TYPE_MAP.inverse[BondType.Single]

            case BondType.Double:
                return MOL2_BOND_TYPE_MAP.inverse[BondType.Double]

            case BondType.Triple:
                return MOL2_BOND_TYPE_MAP.inverse[BondType.Triple]

            case BondType.Aromatic:
                return MOL2_BOND_TYPE_MAP.inverse[BondType.Aromatic]

            case BondType.Amide:
                return MOL2_BOND_TYPE_MAP.inverse[BondType.Amide]

            case BondType.Dummy:
                return MOL2_BOND_TYPE_MAP.inverse[BondType.Dummy]

            case BondType.NotConnected:
                return MOL2_BOND_TYPE_MAP.inverse[BondType.NotConnected]
            
            #Mol2 Format Doesn't Recognize Multi-Attachment Bonds Natively
            case BondType.Ligand:
                return MOL2_BOND_TYPE_MAP.inverse[BondType.Unknown]
            
            case _:
                return MOL2_BOND_TYPE_MAP.inverse[BondType.Unknown]


class Connectivity(Promolecule):
    """This is a parent class that employs methods that work on Promolecule (i.e.
    a list of disconnected atoms with no structure or geometry assigned to them)
    and connections between these disconnected atoms. This can be thought of
    as an undirected graph of nodes (atoms) and edges (bonds) without
    explicit coordinates.
    """

    def __init__(
        self,
        other: Promolecule = None,
        /,
        *,
        n_atoms: int = 0,
        name: str = None,
        copy_atoms: bool = False,
        charge: int = None,
        mult: int = None,
        **kwds,
    ):
        super().__init__(
            other,
            n_atoms=n_atoms,
            name=name,
            copy_atoms=copy_atoms,
            charge=charge,
            mult=mult,
            **kwds,
        )

        if isinstance(other, Connectivity):
            atom_map = {other.atoms[i]: self.atoms[i] for i in range(self.n_atoms)}
            self._bonds = list(
                b.evolve(a1=atom_map[b.a1], a2=atom_map[b.a2], parent=self)
                for b in other.bonds
            )
        else:
            self._bonds = list()

    @property
    def bonds(self) -> List[Bond]:
        """
        Returns
        -------
        List[Bond]
            Returns an ordered list of the Bonds in the Connectivity instance

        Examples
        -------
        The Molecule class inherits bonds
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.bonds
            [Bond(a1=42, a2=22, ...), Bond(a1=41,a2=22, ...), ...]
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> connect.bonds
            [Bond(a1=42, a2=22, ...), Bond(a1=41,a2=22, ...), ...]
        """

        return self._bonds

    @property
    def n_bonds(self) -> int:
        """
        Returns
        -------
        int
            Returns the total number of bonds in the Connectivity instance

        Examples
        -------
        The Molecule class inherits n_bonds
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.n_bonds
            47
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> connect.n_bonds
            47
        """

        return len(self._bonds)

    def lookup_bond(self, a1: AtomLike, a2: AtomLike) -> Bond | None:
        """Retrieves the bond that connects two atoms

        Parameters
        ----------
        a1 : AtomLike
            The first Atom instance
        a2 : AtomLike
            the second Atom instance

        Returns
        -------
        Bond | None
            Returns the bond found or None if the bond doesn't exist

        Examples
        -------
        The Molecule class inherits lookup_bond()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> a1, a2 = dendrobine.get_atoms(0,1)
            >>> dendrobine.lookup_bond(a1, a2)
            Bond(a1=0, a2=1, label=None, ...)
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> connect.lookup_bond(a1, a2)
            Bond(a1=0, a2=1, label=None, ...)
        """

        _a1 = self.get_atom(a1)
        _a2 = self.get_atom(a2)

        try:
            return self.bonds[self.index_bond({_a1, _a2})]
        except:
            return None

    def connect(self, _a1: AtomLike, _a2: AtomLike, **kwds) -> Bond:
        """Connects two atoms together in the same Connectivity instance

        Parameters
        ----------
        _a1 : AtomLike
            The first Atom instance
        _a2 : AtomLike
            The second Atom instance

        Returns
        -------
        Bond
            Returns newly created bond

        Examples
        -------
        The Molecule class inherits connect()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> a1, a2 = dendrobine.get_atoms(0, 35)
            >>> dendrobine.connect(a1, a2, btype=ml.BondType.Single)
            Bond(a1=0, a2=35, label=None, btype=Single, ...)
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> connect.connect(a1, a2, btype=ml.BondType.Single)
            Bond(a1=0, a2=35, label=None, btype=Single, ...)
        """

        a1, a2 = self.get_atoms(_a1, _a2)
        self.append_bond(b := Bond(a1, a2, **kwds))
        return b

    def index_bond(self, b: Bond) -> int:
        """Fetches the atom index from the Connectivity instance

        Parameters
        ----------
        b : Bond
            Must be a bond in the Connectivity instance list

        Returns
        -------
        int
            Returns the index of the bond

        Examples
        -------
        The Molecule class inherits index_bond()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> bond = dendrobine.lookup_bond(3,4)
            >>> dendrobine.index_bond(bond)
            31
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> bond = connect.lookup_bond(3,4)
            >>> connect.index_bond(bond)
            31
        """

        return self._bonds.index(b)

    def get_bond(self, b: Bond | int) -> Bond:
        """Fetches a bond from the Connectivity instance

        Parameters
        ----------
        b : Bond | int
            A bond or index

        Returns
        -------
        Bond
            Returns the Bond instance

        Examples
        -------
        The Molecule class inherits get_bond()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.get_bond(10)
            Bond(a1=21, a2=39, label=None, btype=Single, stereo=Unknown, f_order=1.0)
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> connect.get_bond(10)
            Bond(a1=21, a2=39, label=None, btype=Single, stereo=Unknown, f_order=1.0)
        """

        match b:
            case Bond():
                if b in self._bonds:
                    return b
                else:
                    raise ValueError(f"Bond {b} does not belong to this molecule.")

            case int():
                return self._bonds[b]

            case _:
                raise ValueError(f"Unable to fetch a bond with {type(b)}: {b}")

    def append_bond(self, bond: Bond) -> None:
        """Appends a bond to the Connectivity instance

        Parameters
        ----------
        bond : Bond
            A bond instance to be added

        Examples
        -------
        The Molecule class inherits append_bond()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> a1, a2 = dendrobine.get_atoms(0,35)
            >>> new_bond = ml.Bond(a1,a2)
            >>> dendrobine.append_bond(new_bond)
            >>> dendrobine.lookup_bond(new_bond)
            Bond(a1=0, a2=35, label=None, btype=Single, stereo=Unknown, f_order=1.0)
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> a1, a2 = connect.get_atoms(0,35)
            >>> new_bond = ml.Bond(a1,a2)
            >>> connect.append_bond(new_bond)
            >>> connect.lookup_bond(new_bond)
            Bond(a1=0, a2=35, label=None, btype=Single, stereo=Unknown, f_order=1.0)
        """

        self._bonds.append(bond)
        bond.parent = self

        if bond.a1 not in self.atoms:
            self.append_atom(bond.a1)
        if bond.a2 not in self.atoms:
            self.append_atom(bond.a2)

    def append_bonds(self, *bonds: Bond) -> None:
        """Appends multiple bonds to the Connectivity instance

        Examples
        -------
        The Molecule class inherits append_bonds()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> a1, a2, a3 = dendrobine.get_atoms(0,35, 39)
            >>> b1 = ml.Bond(a1,a2)
            >>> b2 = ml.Bond(a1,a3)
            >>> dendrobine.append_bonds(b1,b2)
            >>> dendrobine.lookup_bond(a1,a2)
            Bond(a1=0, a2=35, label=None, btype=Single, stereo=Unknown, f_order=1.0)
            >>> dendrobine.lookup_bond(a1,a3)
            Bond(a1=0, a2=39, label=None, btype=Single, stereo=Unknown, f_order=1.0)
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> a1, a2, a3 = connect.get_atoms(0,35, 39)
            >>> b1 = ml.Bond(a1,a2)
            >>> b2 = ml.Bond(a1,a3)
            >>> connect.append_bonds(b1,b2)
            >>> connect.lookup_bond(a1,a2)
            Bond(a1=0, a2=35, label=None, btype=Single, stereo=Unknown, f_order=1.0)
            >>> connect.lookup_bond(a1,a3)
            Bond(a1=0, a2=39, label=None, btype=Single, stereo=Unknown, f_order=1.0)
        """
        self._bonds.extend(bonds)
        for b in bonds:
            b.parent = self

            if b.a1 not in self.atoms:
                self.append_atom(b.a1)
            if b.a2 not in self.atoms:
                self.append_atom(b.a2)

    def extend_bonds(self, bonds: Iterable[Bond]) -> None:
        """Extends the list of bonds to the Connectivity instance

        Examples
        -------
        The Molecule class inherits extend_bonds()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> a1, a2, a3 = dendrobine.get_atoms(0,35, 39)
            >>> b1 = ml.Bond(a1,a2)
            >>> b2 = ml.Bond(a1,a3)
            >>> dendrobine.extend_bonds([b1,b2])
            >>> dendrobine.lookup_bond(a1,a2)
            Bond(a1=0, a2=35, label=None, btype=Single, stereo=Unknown, f_order=1.0)
            >>> dendrobine.lookup_bond(a1,a3)
            Bond(a1=0, a2=39, label=None, btype=Single, stereo=Unknown, f_order=1.0)
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> a1, a2, a3 = connect.get_atoms(0,35, 39)
            >>> b1 = ml.Bond(a1,a2)
            >>> b2 = ml.Bond(a1,a3)
            >>> connect.extend_bonds([b1,b2])
            >>> connect.lookup_bond(a1,a2)
            Bond(a1=0, a2=35, label=None, btype=Single, stereo=Unknown, f_order=1.0)
            >>> connect.lookup_bond(a1,a3)
            Bond(a1=0, a2=39, label=None, btype=Single, stereo=Unknown, f_order=1.0)
        """
        self.append_bonds(*bonds)

    def connect_like(self, other: Connectivity):
        """
        Connect the atoms in current instance like they are connected in `other`

        This function assumes that the equivalent atom indices are the same in both sequences.

        Parameters
        ----------
        other : Connectivity
            Instance to copy the connectivity from
        """
        assert self.n_atoms == other.n_atoms, "Atoms must match"
        assert self.elements == other.elements, "Elements must match"

        atom_map = dict(zip(other.atoms, self.atoms))
        self._bonds = [
            b.evolve(a1=atom_map[b.a1], a2=atom_map[b.a2], parent=self)
            for b in other.bonds
        ]

    def del_bond(self, b: Bond) -> None:
        """Deletes a bond from the Connectivity instance

        Parameters
        ----------
        b : Bond
            Bond to delete

        Examples
        -------
        The Molecule class inherits del_bond()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.get_bond(0)
            Bond(a1=42, a2=22, label=None, btype=Single, stereo=Unknown, f_order=1.0)
            >>> dendrobine.del_bond(dendrobine.get_bond(0))
            >>> dendrobine.get_bond(0)
            Bond(a1=41, a2=22, label=None, btype=Single, stereo=Unknown, f_order=1.0)
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> connect.get_bond(0)
            Bond(a1=42, a2=22, label=None, btype=Single, stereo=Unknown, f_order=1.0)
            >>> connect.del_bond(connect.get_bond(0))
            >>> connect.get_bond(0)
            Bond(a1=41, a2=22, label=None, btype=Single, stereo=Unknown, f_order=1.0)
        """

        self._bonds.remove(b)

    def del_atom(self, _a: AtomLike):
        """Deletes an atom and its respective bonds from the Connectivity instance

        Parameters
        ----------
        _a : AtomLike
            An atom, index, label, or Element. This will only delete the first
            instance of the label or Element found

        Examples
        -------
        The Molecule class inherits del_atom()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> print(dendrobine.n_atoms, dendrobine.n_bonds)
            44, 47
            >>> dendrobine.del_atom(0)
            >>> print(dendrobine.n_atoms, dendrobine.n_bonds)
            43, 44
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> print(connect.n_atoms, connect.n_bonds)
            44, 47
            >>> connect.del_bond(connect.get_bond(0))
            >>> print(connect.n_atoms, connect.n_bonds)
            44, 47
        """

        tbd = list(self.bonds_with_atom(_a))
        for b in tbd:
            self.del_bond(b)
        super().del_atom(_a)

    def bonds_with_atom(self, a: AtomLike) -> Generator[Bond, None, None]:
        """Yields bonds attached to an atom in a Connectivity instance

        Parameters
        ----------
        a : AtomLike
            An atom, index, label, or Element. This will only delete the first
            instance of the label or Element found

        Yields
        ------
        Generator[Bond, None, None]
            Yields generator of Bond instances

        Examples
        -------
        The Molecule class inherits bonds_with_atom()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.bonds_with_atom(0)
            <generator object Connectivity.bonds_with_atom at ...>
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> connect.bonds_with_atom(0)
            <generator object Connectivity.bonds_with_atom at ...>
        """
        _a = self.get_atom(a)
        for b in self._bonds:
            if _a in b:
                yield b

    def connected_atoms(self, a: AtomLike) -> Generator[Atom, None, None]:
        """Yields atoms attached to an atom in a Connectivity instance

        Parameters
        ----------
        a : AtomLike
            An atom, index, label, or Element. This will only delete the first
            instance of the label or Element found

        Yields
        ------
        Generator[Atom, None, None]
            Yields generator of Atom instances

        Examples
        -------
        The Molecule class inherits connected_atoms()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.connected_atoms(0)
            <generator object Connectivity.connected_atoms at ...>
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> connect.connected_atoms(0)
            <generator object Connectivity.connected_atoms at ...>
        """

        _a = self.get_atom(a)
        for b in self.bonds_with_atom(_a):
            yield b % _a

    def bonded_valence(self, a: AtomLike) -> float:
        """Sum of valences of the atoms bonded in a Connectivity instance

        Parameters
        ----------
        a : AtomLike
            An atom, index, label, or Element. This will only delete the first
            instance of the label or Element found

        Returns
        -------
        float
            Returns sum of valences of atoms connected

        Examples
        -------
        The Molecule class inherits bonded_valence()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.bonded_valence(0)
            3.0
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> connect.bonded_valence(0)
            3.0
        """

        # TODO: rewrite using sum()
        _a_bonds = self.bonds_with_atom(a)

        val = 0.0
        for b in _a_bonds:
            val += b.order

        return val

    def n_bonds_with_atom(self, a: AtomLike) -> int:
        """Total number of bonds to an atom in a Connectivity instance

        Parameters
        ----------
        a : AtomLike
            An atom, index, label, or Element. This will only delete the first
            instance of the label or Element found

        Returns
        -------
        int
            Returns the total number of bonds to an atom

        Examples
        -------
        The Molecule class inherits n_bonds_with_atom()
            >>> dendrobine = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
            >>> dendrobine.n_bonds_with_atom(0)
            3
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dendrobine)
            >>> connect.n_bonds_with_atom(0)
            3
        """

        return sum(1 for _ in self.connected_atoms(a))

    # def _bfs_single(self, q: deque, visited: set):
    #     start, dist = q.pop()
    #     for a in self.connected_atoms(start):
    #         if a not in visited:
    #             yield (a, dist + 1)
    #             visited.add(a)
    #             q.appendleft((a, dist + 1))

    def yield_bfsd(
        self,
        _start: AtomLike,
        _direction: AtomLike = None,
    ) -> Generator[Tuple[Atom, int], None, None]:
        """Yields atoms in breadth-first search, in traversal order,
        Distance from the start atom is also yielded

        Parameters
        ----------
        _start : AtomLike
            An atom, index, label, or Element. This will only delete the first
            instance of the label or Element found
        _direction : AtomLike, optional
            An atom, index, label, or Element. This will only delete the first
            instance of the label or Element found, by default None

        Yields
        ------
        Generator[Tuple[Atom, int], None, None]
            the atoms in breadth-first search, in traversal order,
            Distance from the start atom is also yielded

        Examples
        -------
        The Molecule class inherits yield_bfsd()
            >>> dmf = ml.Molecule.load_mol2(ml.files.dmf_mol2)
            >>> dmf.yield_bfsd(0)
            <generator object Connectivity.yield_bfsd at ...>
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dmf)
            >>> connect.yield_bfsd(0)
            <generator object Connectivity.yield_bfsd at ...>
        """

        start = self.get_atom(_start)
        visited = {start}
        queue = deque()

        if _direction is None:
            queue.append((start, 0))
            # yield (start, 0)
        else:
            direction = self.get_atom(_direction)
            assert direction in set(
                self.connected_atoms(start)
            ), "Direction must be an atom connected to the start"
            visited.add(direction)
            queue.append((direction, 1))
            yield (direction, 1)
        while queue:
            start, dist = queue.pop()
            for a in self.connected_atoms(start):
                if a not in visited:
                    yield (a, dist + 1)
                    visited.add(a)
                    queue.appendleft((a, dist + 1))

    def yield_bfs(
        self, _start: AtomLike, _direction: AtomLike = None
    ) -> Generator[Atom, None, None]:
        """Yields atoms in breadth-first search, in traversal order,
        Distance is not yielded

        Parameters
        ----------
        _start : AtomLike
            An atom, index, label, or Element. This will only delete the first
            instance of the label or Element found
        _direction : AtomLike, optional
            An atom, index, label, or Element. This will only delete the first
            instance of the label or Element found, by default None

        Yields
        ------
        Generator[Atom, None, None]
            The Atoms in a breadth-first search, in traversal order

        Examples
        -------
        The Molecule class inherits yield_bfsd()
            >>> dmf = ml.Molecule.load_mol2(ml.files.dmf_mol2)
            >>> dmf.yield_bfsd(0)
            <generator object Connectivity.yield_bfs at ...>
        If desired, one can work directly with Connectivity class instead
            >>> connect = ml.Connectivity(dmf)
            >>> connect.yield_bfsd(0)
            <generator object Connectivity.yield_bfs at ...>
        """

        start = self.get_atom(_start)
        visited = {start}
        queue = deque()

        if _direction is None:
            queue.append(start)
            # yield (start, 0)
        else:
            direction = self.get_atom(_direction)
            assert direction in set(
                self.connected_atoms(start)
            ), "Direction must be an atom connected to the start"
            visited.add(direction)
            queue.append(direction)
            yield direction
        while queue:
            start = queue.pop()
            for a in self.connected_atoms(start):
                if a not in visited:
                    yield a
                    visited.add(a)
                    queue.appendleft(a)

    def is_bond_in_ring(self, _b: Bond):
        connections = {a for a in self.connected_atoms(_b.a1) if a != _b.a2}
        for a in self.yield_bfs(_b.a1, _b.a2):
            if a in connections:
                return True

        return False

    def to_nxgraph(self) -> nx.Graph:
        """
        Converts an insantce of Connectivity class into networkx object

        Returns:
        --------
        nx_mol: nx.Graph()
            instance of Networkx Graph object
        Notes:
        ------
        In latest version, all the atom and bond attributes are added to Networkx Graph.
        """
        # older working version:
        # with ml.aux.timeit("Converting molli Connectivity into networkx object"):
        # nx_mol = nx.Graph()
        # for atom in self.atoms:
        #     nx_mol.add_node(atom, element=atom.element, label=atom.label, isotope=atom.isotope)
        # for bond in self.bonds:
        #     nx_mol.add_edge(bond.a1, bond.a2, order = bond.order)

        nx_mol = nx.Graph()
        for atom in self.atoms:
            nx_mol.add_node(atom, **atom.as_dict())  # recursion?

        # TODO: re-run test examples

        # nx_mol.add_nodes_from(self.atoms, **self.atom.as_dict())

        for bond in self.bonds:
            nx_mol.add_edge(bond.a1, bond.a2, **bond.as_dict())

        # didn't work: b should be 3-tuple
        # nx_mol.add_edges_from(
        #     [b.as_tuple() for b in self.bonds]  # , **self.bonds[0].as_dict()
        # )
        return nx_mol

    def find_cycle_containing_atom(self, start: AtomLike) -> list:
        """
        Finds the first cycle containing "start" atom
        Parameters:
        -----------
        start: ml.chem.AtomLike
            atom or its atomic index or a unique identifier for starting searching for cycles (loops)
        Returns:
        --------
        cycle: list
            the first found cycle that countains "start" atom

        Current implementation using networkx grpah. Should be rewritten w/o any extra dependencies.
        """
        atom = next(self.yield_atoms_by_element(start))
        nx_mol = self.to_nxgraph()

        for cycle in nx.cycle_basis(nx_mol, atom):
            if atom in cycle:
                return cycle

    @staticmethod
    def _node_match(a1: dict, a2: dict) -> bool:
        # print({x: a1[x] == a2[x] for x in a1 if x in a2})
        """
        Callable helper function that compares attributes of the nodes(atoms) in nx.isomorphism.GraphMatcher().
        Returns True is nodes(atoms) are considered equal, False otherwise.
        For further information: refer to nx.isomorphism.GraphMatcher() documentation.
        """

        # TODO: which attributes might be query?

        if a2["element"] != Element.Unknown and a1["element"] != a2["element"]:
            # print("element:", a1["element"], a2["element"])
            return False

        if (
            a2["isotope"] is not None
            # and a2["isotope"] is not None  # think about it
            and a1["isotope"] != a2["isotope"]
        ):
            # print("isotope:", a1["isotope"], a2["isotope"])
            return False

        # NOTE "geom" and "label" are not compared

        if a2["stereo"] != AtomStereo.Unknown and a1["stereo"] != a2["stereo"]:
            # NOTE: no queries for now
            # print("stereo:", a1["stereo"], a2["stereo"])
            return False

        if a1["atype"] != AtomType.Unknown and a2["atype"] != a2["atype"]:
            # TODO: add groups for queries
            # print("atype:", a1["atype"], a2["atype"])
            return False

        return True

    @staticmethod
    def _edge_match(e1: dict, e2: dict) -> bool:  # TODO: needs improving!
        """
        Callable helper function that compares attributes of the edges(bonds) in nx.isomorphism.GraphMatcher().
        Returns True is edges(bonds) are considered equal, False otherwise.
        For further information: refer to nx.isomorphism.GraphMatcher() documentation.
        """
        # print({x: e1[x] == e2[x] for x in e1 if x in e2})

        match e2["btype"]:
            case BondType.Unknown:
                pass
            case BondType.Single | BondType.Double | BondType.Triple:
                # should work for aromatic and resonating structures
                if e1["btype"] < e2["btype"]:
                    # print("btype 1-3:", e1["btype"], e2["btype"])
                    return False
            case BondType.Aromatic | BondType.Amide:
                if e1["btype"] != e2["btype"]:
                    # print("btype aromatic or amide:", e1["btype"], e2["btype"])
                    return False
            case BondType.NotConnected:
                # print("bond not connected")
                return False
            case BondType.Dummy | _:
                raise NotImplementedError

        match e2["stereo"]:
            case BondStereo.Unknown:
                pass
            case _:
                if e1["stereo"] != e2["stereo"]:
                    # print("bond stereo not equal")
                    return False

        match e2["label"]:
            case None:
                pass
            case _:
                if e1["label"] != e2["label"]:
                    # print("blabel:", e1["label"], e2["label"])
                    return False

        return True

    @staticmethod
    def _edge_match_debug(e1: dict, e2: dict):
        res = Connectivity._edge_match(e1, e2)
        print("Bonds")
        print(f"{e1}\n{e2}\n{res}\n")
        # print(e1["btype"], e2["btype"], res)
        return res

    @staticmethod
    def _node_match_debug(a1, a2):
        res = Connectivity._node_match(a1, a2)
        print("Atoms")
        print(f"{a1}\n{a2}\n{res}\n")
        # print(a1["atype"], a2["atype"], res)
        return res

    def match(
        self,
        pattern: Connectivity,
        /,
        *,
        node_match: Callable[[dict, dict], bool] | None = None,
        edge_match: Callable[[dict, dict], bool] | None = None,
    ) -> Generator[dict | Any, None, None]:
        # TODO: add new parameters to docs
        """
        Checks two molli connectivities for isomorphism.
        Yields generator over subgraph isomorphism mappings.

        ```python
        for mapping in connectivity.match(pattern):
            ...
        ```
        Parameters:
        -----------
        pattern: Connectivity
            query-Connectivity (Molecule) to match with given Connectivity
        """
        nx_pattern = pattern.to_nxgraph()
        nx_source = self.to_nxgraph()

        if not node_match:
            node_match = self._node_match
        if not edge_match:
            edge_match = self._edge_match

        matcher = nx.isomorphism.GraphMatcher(
            nx_source,
            nx_pattern,
            node_match=node_match,
            edge_match=edge_match,
        )

        for ismorphism in matcher.subgraph_isomorphisms_iter():
            yield {v: k for k, v in ismorphism.items()}

    def get_substr_indices(
        self, pattern: Connectivity
    ) -> Generator[list[int], None, None]:
        """
        Yields all possible combinations of substructure indices that matched
        with the given pattern.

        Parameters:
        -----------
        pattern: Connectivity

        Returns:
        --------
        Generator over list of all possible mappings to pattern

        If only one variation of substructure indices is needed, use
        next(ens.get_substr_indices(pattern))

        ``python
        for ens in tqdm(library):
            for mapping in ens.get_substr_indices(pattern):
                ...
        ```
        """
        mappings = self.match(
            pattern,
            node_match=Connectivity._node_match,
            edge_match=Connectivity._edge_match,
        )
        atom_idx = {a: i for i, a in enumerate(self.atoms)}

        for mapping in mappings:
            yield [atom_idx[mapping[x]] for x in pattern.atoms]

    def connect(self, _a1: AtomLike, _a2: AtomLike, **kwds):
        a1, a2 = self.get_atoms(_a1, _a2)
        self.append_bond(b := Bond(a1, a2, **kwds))
        return b
