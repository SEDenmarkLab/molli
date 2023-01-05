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

class BondType(IntEnum):
    Unknown = 0
    Single = 1
    Double = 2
    Triple = 3
    Quadruple = 4
    Quintuple = 5
    Sextuple = 6

    FractionalOrder = 10

    Aromatic = 20
    Amide = 30

    H_Donor = 100
    H_Acceptor = 101

class BondStereo(IntEnum):
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


@attrs.define(slots=True, repr=True, hash=False, eq=False, weakref_slot=True)
class Bond:
    """`a1` and `a2` are always assumed to be interchangeable"""

    a1: Atom
    a2: Atom

    label: str = attrs.field(default=None, kw_only=True)
    f_order: float = attrs.field(default=1.0, converter=float, kw_only=True)
    btype: BondType = attrs.field(
        default=BondType.Single,
        kw_only=True,
        repr=lambda x: x.name,
    )
    stereo: BondStereo = attrs.field(
        default=BondStereo.Unknown,
        kw_only=True,
        repr=lambda x: x.name,
    )

    mol2_type: str = attrs.field(default="1", kw_only=True, repr=False)

    def evolve(self, **changes):
        return attrs.evolve(self, **changes)

    @property
    def order(self) -> float:
        # if self.btype == BondType.FractionalOrder:
        #     return self._order
        # elif 0 < self.btype < 9:
        #     return float(self.btype)
        # elif self.btype in {BondType.Aromatic}

        match self.btype:

            case 0|1|2|3|4|5|6 as b:
                return float(b)
            
            case BondType.Aromatic:
                return 1.5

            case BondType.FractionalOrder:
                return self.f_order
            
            case BondType.H_Acceptor:
                return 0.0
            
            case _:
                return 1.0
    
    def copy(self, other: Bond):
        for f in Bond.__slots__:
            setattr(self, f, getattr(other, f))

    def __contains__(self, other: Atom):
        return other in {self.a1, self.a2}

    def __eq__(self, other: Bond | set):
        # return self is other
        match other:
            case Bond():
                return {self.a1, self.a2} == {other.a1, other.a2}
            case list()|set() as l:
                a1, a2 = l
                return {self.a1, self.a2} == {a1, a2}
            case _:
                raise ValueError(
                    f"Cannot equate <{type(other)}: {other}>, {self}"
                )
    

    def __repr__(self):
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
        return self.a1.cov_radius_1 + self.a2.cov_radius_1
       


class Connectivity(Promolecule):
    def __init__(
        self, other: Promolecule = None, /, *, n_atoms: int = 0, name="unnamed", copy_atoms: bool = False, **kwds, 
    ):
        super().__init__(other, n_atoms=n_atoms, name=name, copy_atoms=copy_atoms, **kwds)

        if isinstance(other, Connectivity):
            atom_map = {other.atoms[i] : self.atoms[i] for i in range(self.n_atoms)}
            self._bonds = list(b.evolve(a1 = atom_map[b.a1], a2 = atom_map[b.a2]) for b in other.bonds)
        else:
            self._bonds = list()
        


    @property
    def bonds(self) -> List[Bond]:
        return self._bonds

    @property
    def n_bonds(self):
        return len(self._bonds)

    def lookup_bond(self, a1: AtomLike, a2: AtomLike):
        """
        Returns a bond that connects the two atoms. O(N).
        NOTE: `a1` and `a2` are always assumed to be interchangeable in this context.
        """
        _a1 = self.get_atom(a1)
        _a2 = self.get_atom(a2)

        try:
            return self.bonds[self.index_bond({_a1, _a2})]
        except:
            return None 

    def index_bond(self, b: Bond) -> int:
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
        self._bonds.append(bond)

    def append_bonds(self, *bonds: Bond):
        self._bonds.extend(bonds)
    
    def extend_bonds(self, bonds: Iterable[Bond]):
        self._bonds.extend(bonds)
    

    def del_bond(self, b: Bond):
        match b:
            case Bond():
                idx = self.index_atom(b)
            case int():
                idx = b
            case _:
                raise ValueError("Cannot delete a bond with this argument")

        del self._bonds[idx]

    def bonds_with_atom(self, a: AtomLike) -> Generator[Bond, None, None]:
        _a = self.get_atom(a)
        for b in self._bonds:
            if _a in b:
                yield b

    def connected_atoms(self, a: AtomLike) -> Generator[Atom, None, None]:
        _a = self.get_atom(a)
        for b in self.bonds_with_atom(_a):
            yield b % _a

    def bonded_valence(self, a: AtomLike):
        _a_bonds = self.bonds_with_atom(a)

        val = 0.0
        for b in _a_bonds:
            val += b.order

        return val

    def _bfs_single(self, q: deque, visited: set):
        start, dist = q.pop()
        for a in self.connected_atoms(start):
            if a not in visited:
                yield (a, dist + 1)
                visited.add(a)
                q.appendleft((a, dist + 1))

    def yield_bfsd(self, start: AtomLike) -> Generator[Tuple[Atom, int], None, None]:
        """
        Yields atoms and their distances from start

        ```python
        for atom, distance in connectivity.yield_bfsd(a):
            ...
        ```
        """
        _sa = self.get_atom(start)
        visited = set((_sa,))
        q = deque([(_sa, 0)])
        while q:
            yield from self._bfs_single(q, visited)


    