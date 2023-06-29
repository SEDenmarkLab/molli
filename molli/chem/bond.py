from __future__ import annotations
from . import Atom, AtomLike, Promolecule, PromoleculeLike
from dataclasses import dataclass, field, KW_ONLY
from typing import Iterable, List, Generator, Tuple, Any
from copy import deepcopy
from enum import IntEnum
from collections import deque
from struct import pack, unpack, Struct
from io import BytesIO
import attrs
from bidict import bidict
from functools import cache


class BondType(IntEnum):
    """# `BondType`
    
    Enumerates through different kinds of bonds and assignes hydrogen donor and acceptorclassifications

    Args:

        IntEnum (cls): inherited class for enumeration of integers

    Potential Uses:
    ```Python
        double_bond = BondType(2) # Double
        donor = BondType(100) # hydrogen donor
        donor == BondType.H_Donor # True 
    ```
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

    FractionalOrder = 99

    H_Donor = 100
    H_Acceptor = 101


class BondStereo(IntEnum):
    """
    Enumerates through different classifications for bond stereochemistry 

    Example Usage:
    
        >>> trans = BondStereo(10) # E
        >>> trans == BondStereo.E # True 
        >>> trans == BondStereo.Trans # True 
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
        # "4" : BondType.Quadruple,
        # "5" : BondType.Quintuple,
        # "6" : BondType.Sextuple,
        "ar": BondType.Aromatic,
        "am": BondType.Amide,
        "du": BondType.Dummy,
        "un": BondType.Unknown,
        "nc": BondType.NotConnected,
    }
)


@attrs.define(slots=True, repr=True, hash=False, eq=False, weakref_slot=True)
class Bond:
    """
    The class for bonds in the MOLLI package.
    a1 and a2 are the atoms that the bond connects, and their order is assumed to be interchangeable.
    """    
    a1: Atom
    a2: Atom

    label: str = attrs.field(
        default=None,
    )

    btype: BondType = attrs.field(
        default=BondType.Single,
        repr=lambda x: x.name,
    )

    stereo: BondStereo = attrs.field(
        default=BondStereo.Unknown,
        repr=lambda x: x.name,
    )

    f_order: float = attrs.field(
        default=1.0,
        converter=float,
    )

    def evolve(self, **changes) -> Bond:
        """
        Creates a new bond with the same atoms, but with the specified changes

        Args:
            **changes: the changes to make to the bond class
        
        Returns:
            Bond: the new bond with the specified changes
        
        Example Usage: 
            >>> bond = Bond(a1, a2, order=2)
            >>> bond.evolve(order=3) # Bond(a1, a2, order=3)
        """
        return attrs.evolve(self, **changes)

    @property
    def order(self) -> float:
        """
        The order of the bond

        Returns:
            float: the order of the bond
        
        Example Usage:
            >>> bond = Bond(a1 = "C", a2 = "H")
            >>> print(bond.order) # 1.0
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

            case BondType.Dummy | BondType.NotConnected:
                return 0.0

            case _:
                return 1.0

    def as_dict(self, atom_id_map: dict = None) -> dict:
        """
        Converts the bond to a dictionary
    
        Args:
            atom_id_map (dict): a dictionary mapping atom ids to atom indices
        
        Returns:
            dict: the bond as a dictionary
        
        Example Usage:
            >>> bond = Bond(a1 = "C", a2 = "H")
            >>> bond.as_dict({0: 'C', 1: 'H'}) 
            >>> print(bond) # {0: 'C', 1: 'H'}
        """
        res = attrs.asdict(self)
        if atom_id_map is not None:
            res["a1"] = atom_id_map[self.a1]
            res["a2"] = atom_id_map[self.a2]
        return res

    def as_tuple(self, atom_id_map: dict = None) -> tuple:
        """
        Converts the bond to a tuple
    
        Args:
            atom_id_map (dict): a tuple mapping atom ids to atom indices
        
        Returns:
            tuple: the bond as a tuple
        
        Example Usage:
            >>> bond.as_tuple({0: "C", 1:"H"}) 
            >>> print(bond) # ('C', 'H')
        """
        res = attrs.astuple(self)

        if atom_id_map is not None:
            a1, a2, *rest = res
            res = (atom_id_map[self.a1], atom_id_map[self.a2], *rest)

        return res

    def __contains__(self, other: Atom) -> bool:
        """
        boolean check if an atom is in the bond
        
        Args:
            other (Atom): the atom to check
        
        Returns:
            bool: True if the atom is in the bond, False otherwise
        
        Example Usage:
            >>> bond = Bond(a1 = "C", a2 = "H")
            >>> bond.__contains__("C") # TRUE 
        """
        return other in {self.a1, self.a2}

    def __eq__(self, other: Bond | set):
        """
        Boolean check if two bonds are equal
        
        Args: 
            other (Bond | set): the bond to check
        
        Returns:
            bool: True if the bonds are equal, False otherwise
        
        Example Usage:
            >>> bond_1 = Bond(a1 = "C", a2 = "H")
            >>> bond_2 = Bond(a1 = "C", a2 = "H")
            >>> bond_1 == bond_2 # True
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

    def __repr__(self):
        """
        Prints the bond

        Returns:
            str: the string representation of the bond
        
        Example Usage:
            >>> bond = Bond(a1 = "C", a2 = "H")
            >>> bond # Bond(C, H, order=1.0)
        """
        return f"Bond({self.a1}, {self.a2}, order={self.order})"

    # This mimics the default behavior of object instances in python
    def __hash__(self) -> int:
        return id(self) >> 4

    def __mod__(self, a: Atom) -> Atom:
        if self.a1 == a:
            return self.a2
        elif self.a2 == a:
            return self.a1
        else:
            raise ValueError("Atom is not a part of this bond")

    @property
    def expected_length(self) -> float:
        """
        Calculates bond length in Angstroms based on the covalent radii of the atoms.

        Returns:
            float: the expected bond length in Angstroms
        
        Example Usage:
            >>> bond = Bond(a1 = "H", a2 = "C")
            >>> print(bond.expected_length) # 1.09
        """        
        return self.a1.cov_radius_1 + self.a2.cov_radius_1

    @cache
    def set_mol2_type(self, m2t: str):
        self.btype = MOL2_BOND_TYPE_MAP[m2t]

    def get_mol2_type(self):
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


class Connectivity(Promolecule):
    """
    Connectivity is a graph-like structure that connects the nodes(atoms) with edges(bonds).

    Args:
        Promolecule (cls): inherited class for molecules
    """
    
    # __slots__ = "_atoms", "_bonds", "_name", "charge", "mult"

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
                b.evolve(a1=atom_map[b.a1], a2=atom_map[b.a2]) for b in other.bonds
            )
        else:
            self._bonds = list()

    @property
    def bonds(self) -> List[Bond]:
        """
        List of all bonds in the molecule

        Returns:
            List[Bond]: list of all bonds in the molecule
        
        Example Usage:
            >>> molecule = Connectivity('Acetyl Chloride') # CH3COCl
            >>> print(molecule.bonds) # [Bond(C, H, order=1.0), Bond(C, C, order=1.0), Bond(C, O, order=2.0), Bond(C, Cl, order=1.0)]
        """
        return self._bonds

    @property
    def n_bonds(self) -> int:
        """
        The total number of bonds in the molecule
        
        Returns:
            int: the total number of bonds in the molecule
        
        Example Usage:
            >>> molecule = Connectivity('Acetyl Chloride') # CH3COCl
            >>> print(molecule.n_bonds) # 7
        """        
        return len(self._bonds)

    def lookup_bond(self, a1: AtomLike, a2: AtomLike) -> Bond | None:
        """
        Retrieves the bond that connects two atoms 

        Args:
            a1 (AtomLike): the first atom
            a2 (AtomLike): the second atom
        
        Returns:
            Bond | None: the bond that connects the two atoms, or None if the bond does not exist
        
        Example Usage:
            >>> molecule = Connectivity('Acetyl Chloride') # CH3COCl
            >>> print(molecule.lookup_bond("C", "H")) # Bond(C, H, order=1.0)
        """        
        _a1 = self.get_atom(a1)
        _a2 = self.get_atom(a2)

        try:
            return self.bonds[self.index_bond({_a1, _a2})]
        except:
            return None

    def index_bond(self, b: Bond) -> int:
        """
        Index of Desired Bond 

        Args:
            b (Bond): the bond to find the index of
        Returns:
            int: the index of the bond
        
        Example Usage:
            >>> molecule = Connectivity('Acetyl Chloride') # CH3COCl
            >>> print(molecule.index_bond(Bond("C", "H"))) # 0
        """        
        return self._bonds.index(b)

    def get_bond(self, b: Bond | int) -> Bond:
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

    def append_bond(self, bond: Bond):
        """
        Adds a bond to the molecule

        Args:
            bond (Bond): the bond to add to the molecule
        
        Example Usage:
            >>> molecule = Connectivity('Acetyl Chloride') # CH3COCl
            >>> molecule.append_bond(Bond("C", "H"))
            >>> print(molecule.n_bonds) # 8
        """
        self._bonds.append(bond)

    def append_bonds(self, *bonds: Bond):
        """
        Adds multiple bonds to the molecule

        Args: 
            *bonds (Bond): the bonds to add to the molecule
        
        Example Usage:
            >>> molecule = Connectivity('Acetyl Chloride') # CH3COCl
            >>> molecule.append_bonds(Bond("C", "H"), Bond("C", "H"))
            >>> print(molecule.n_bonds) # 9
        """
        self._bonds.extend(bonds)

    def extend_bonds(self, bonds: Iterable[Bond]):
        """
        Adds multiple bonds to the molecule

        Args: 
            *bonds (Bond): the bonds to add to the molecule
        
        Example Usage:
            >>> molecule = Connectivity('Acetyl Chloride') # CH3COCl
            >>> molecule.append_bonds(Bond("C", "H"), Bond("C", "H"))
            >>> print(molecule.n_bonds) # 9
        """
        self._bonds.extend(bonds)

    def del_bond(self, b: Bond):
        """
        Deletes a bond from the molecule

        Args:
            b (Bond): the bond to delete from the molecule
        
        Example Usage:
            >>> molecule = Connectivity('Acetyl Chloride') # CH3COCl
            >>> molecule.del_bond(Bond("C", "H"))
            >>> print(molecule.n_bonds) # 6
        """
        self._bonds.remove(b)

    def del_atom(self, _a: AtomLike):
        """
        Deletes an atom from the molecule

        Args:
            _a (AtomLike): the atom to delete from the molecule
        
        Example Usage:
            >>> molecule = Connectivity('Acetyl Chloride') # CH3COCl
            >>> molecule.del_atom("C")
            >>> print(molecule.get_atom("C")) # [Atom(H), Atom(O), Atom(Cl)]
        """
        tbd = list(self.bonds_with_atom(_a))
        for b in tbd:
            self.del_bond(b)
        super().del_atom(_a)

    def bonds_with_atom(self, a: AtomLike) -> Generator[Bond, None, None]:
        """
        Each bond on an atom

        Args:
            a (AtomLike): the atom to find the bonds of        
        
        Yields:
            Generator[Bond, None, None]: the bonds on the atom
        
        Example Usage:
            >>> molecule = Connectivity('Acetyl Chloride') # CH3COCl
            >>> print(molecule.bonds_with_atom("C")) # [Bond(C, H, order=1.0), Bond(C, C, order=1.0), Bond(C, O, order=2.0), Bond(C, Cl, order=1.0)]
        """        
        _a = self.get_atom(a)
        for b in self._bonds:
            if _a in b:
                yield b

    def connected_atoms(self, a: AtomLike) -> Generator[Atom, None, None]:
        """
        Each atom connected to an atom

        Args:
            a (AtomLike): the atom to find the connected atoms of
        
        Yields:
            Generator[Atom, None, None]: the atoms connected to the atom

        Example Usage:
            >>> molecule = Connectivity('Acetyl Chloride') # CH3COCl
            >>> print(molecule.connected_atoms("C")) # [Atom(C), Atom(H), Atom(C), Atom(O), Atom(Cl)]
        """
        _a = self.get_atom(a)
        for b in self.bonds_with_atom(_a):
            yield b % _a

    def bonded_valence(self, a: AtomLike) -> float:
        """
        Sum of valences of all atoms bonded to an atom

        Args:
            a (AtomLike): the atom to find the bonded valence of
        
        Returns:
            float: the sum of valences of all atoms bonded to the atom
        
        Example Usage:
            >>> molecule = Connectivity('Acetyl Chloride') # CH3COCl
            >>> print(molecule.bonded_valence("C")) # 4.0
        """
        _a_bonds = self.bonds_with_atom(a)

        val = 0.0
        for b in _a_bonds:
            val += b.order

        return val

    def n_bonds_with_atom(self, a: AtomLike) -> int:
        """
        Total number of bonds on an atom
        
        Args:
            a (AtomLike): the atom to find the number of bonds of
        
        Returns:
            int: the total number of bonds on the atom
        """        
        return sum(1 for _ in self.connected_atoms(a))

    def _bfs_single(self, q: deque, visited: set) -> Generator[Tuple[Atom, int], None, None]:
        start, dist = q.pop()
        for a in self.connected_atoms(start):
            if a not in visited:
                yield (a, dist + 1)
                visited.add(a)
                q.appendleft((a, dist + 1))

    def yield_bfsd(self, start: AtomLike) -> Generator[Tuple[Atom, int], None, None]:
        """
        Yields atoms in breadth-first search, in traversal order.
        
        Distance from the start atom is also yielded.

        Args:
            start (AtomLike): the atom to start the search from
        
        Yields:
            Generator[Tuple[Atom, int], None, None]: the atoms in breadth-first search, in traversal order
        """
        _sa = self.get_atom(start)
        visited = set((_sa,))
        q = deque([(_sa, 0)])
        while q:
            yield from self._bfs_single(q, visited)
