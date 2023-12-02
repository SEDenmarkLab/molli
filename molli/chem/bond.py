from __future__ import annotations
from . import Atom, Element, AtomLike, Promolecule, PromoleculeLike
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
from weakref import ref, WeakKeyDictionary, WeakSet


class BondType(IntEnum):
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
    """`a1` and `a2` are always assumed to be interchangeable"""

    a1: Atom = attrs.field(repr=lambda a: a.idx or a)
    a2: Atom = attrs.field(repr=lambda a: a.idx or a)

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

    attrib: dict = attrs.field(factory=dict, repr=False)

    _parent = attrs.field(
        default=None,
        repr=False,
        # converter=lambda x: None if x is None else ref(x),
    )

    @property
    def parent(self):
        if self._parent is None:
            return None
        else:
            return self._parent()

    @parent.setter
    def parent(self, other):
        self._parent = ref(other)

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

    def as_dict(self, schema: List[str] = None):
        if schema is None:
            return attrs.asdict(self)
        else:
            return {a: getattr(self, a, None) for a in schema}

    def as_tuple(self, schema: List[str] = None):
        if schema is None:
            return attrs.astuple(self)
        else:
            return tuple(getattr(self, a, None) for a in schema)

    def __contains__(self, other: Atom):
        return other in {self.a1, self.a2}

    def __eq__(self, other: Bond | set):
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
        r1 = self.a1.cov_radius_1
        r2 = self.a2.cov_radius_1
        return (r1 or Element.C.cov_radius_1) + (r2 or Element.C.cov_radius_1)

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

    def connect(self, _a1: AtomLike, _a2: AtomLike, **kwds):
        a1, a2 = self.get_atoms(_a1, _a2)
        self.append_bond(b := Bond(a1, a2, **kwds))
        return b

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
        bond.parent = self

    def append_bonds(self, *bonds: Bond):
        self._bonds.extend(bonds)
        for b in bonds:
            b.parent = self

    def extend_bonds(self, bonds: Iterable[Bond]):
        self._bonds.extend(bonds)
        for b in bonds:
            b.parent = self

    def del_bond(self, b: Bond):
        self._bonds.remove(b)

    def del_atom(self, _a: AtomLike):
        tbd = list(self.bonds_with_atom(_a))
        for b in tbd:
            self.del_bond(b)
        super().del_atom(_a)

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
        # TODO: rewrite using sum()
        _a_bonds = self.bonds_with_atom(a)

        val = 0.0
        for b in _a_bonds:
            val += b.order

        return val

    def n_bonds_with_atom(self, a: AtomLike):
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
        """
        Yields atoms and their distances from start

        ```python
        for atom, distance in connectivity.yield_bfsd(a):
            ...
        ```
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
        """
        Yields atoms and their distances from start

        ```python
        for atom in connectivity.yield_bfsd(a):
            ...
        ```
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
