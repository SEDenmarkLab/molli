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
import networkx as nx


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

    def as_dict(self, atom_id_map: dict = None):
        res = attrs.asdict(self)
        if atom_id_map is not None:
            res["a1"] = atom_id_map[self.a1]
            res["a2"] = atom_id_map[self.a2]
        return res

    def as_tuple(self, atom_id_map: dict = None):
        res = attrs.astuple(self)

        if atom_id_map is not None:
            a1, a2, *rest = res
            res = (atom_id_map[self.a1], atom_id_map[self.a2], *rest)

        return res

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
                b.evolve(a1=atom_map[b.a1], a2=atom_map[b.a2]) for b in other.bonds
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

    def connect(self, _a1: AtomLike, _a2: AtomLike, **kwds):
        a1, a2 = self.get_atoms(_a1, _a2)
        self.append_bond(b := Bond(a1, a2, **kwds))
        return b
