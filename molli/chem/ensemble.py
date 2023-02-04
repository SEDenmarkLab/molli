from __future__ import annotations
from typing import Iterable, Iterator, List, Callable
from . import (
    Molecule,
    Atom,
    Element,
    Promolecule,
    Bond,
    Connectivity,
    CartesianGeometry,
    Structure,
    StructureLike,
    PromoleculeLike,
)

# from .geometry import _nans
from .structure import RE_MOL_NAME, RE_MOL_ILLEGAL
from ..parsing import read_mol2

import numpy as np
from numpy.typing import ArrayLike
from warnings import warn
from io import StringIO
from pathlib import Path


class ConformerEnsemble(Connectivity):
    def __init__(
        self,
        other: ConformerEnsemble = None,
        /,
        n_conformers: int = 0,
        n_atoms: int = 0,
        *,
        name: str = None,
        charge: int = None,
        mult: int = None,
        coords: ArrayLike = None,
        weights: ArrayLike = None,
        atomic_charges: ArrayLike = None,
        copy_atoms: bool = False,
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

        self._coords = np.full((n_conformers, self.n_atoms, 3), np.nan)
        self._atomic_charges = np.zeros((self.n_atoms,))
        self._weights = np.ones((n_conformers,))

        if isinstance(other, ConformerEnsemble):
            self.atomic_charges = atomic_charges
            self.coords = other.coords
            self.weights = other.weights
        else:
            if coords is not None:
                self.coords = coords

            if atomic_charges is not None:
                self.atomic_charges = atomic_charges

            if weights is not None:
                self.weights = weights

    @property
    def coords(self):
        """Set of atomic positions in shape (n_atoms, 3)"""
        return self._coords

    @coords.setter
    def coords(self, other: ArrayLike):
        self._coords[:] = other

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, other: ArrayLike):
        self._weights[:] = other

    @property
    def atomic_charges(self):
        return self._atomic_charges

    @atomic_charges.setter
    def atomic_charges(self, other: ArrayLike):
        self._atomic_charges[:] = other

    @classmethod
    def from_mol2(
        cls: type[ConformerEnsemble],
        input: str | StringIO,
        /,
        name: str = None,
        charge: int = None,
        mult: int = None,
        source_units: str = "Angstrom",
    ) -> ConformerEnsemble:
        """ """
        mol2io = StringIO(input) if isinstance(input, str) else input
        mols = Molecule.load_all_mol2(mol2io, name=name, source_units=source_units)

        res = cls(
            mols[0],
            n_conformers=len(mols),
            n_atoms=mols[0].n_atoms,
            name=mols[0].name,
            charge=charge,
            mult=mult,
            atomic_charges=mols[0].atomic_charges,
        )

        for i, m in enumerate(mols):
            res._coords[i] = m.coords

        return res

    @classmethod
    def load_mol2(
        cls: type[ConformerEnsemble],
        input: str | Path | IO,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> ConformerEnsemble:
        """Load mol2 from a file stream or file"""
        if isinstance(input, str | Path):
            stream = open(input, "rt")
        else:
            stream = input

        with stream:
            res = cls.from_mol2(
                stream,
                name=name,
                source_units=source_units,
            )

        return res

    @classmethod
    def loads_mol2(
        cls: type[ConformerEnsemble],
        input: str,
        *,
        name: str = None,
        source_units: str = "Angstrom",
    ) -> ConformerEnsemble:
        """Load mol2 file from string"""
        stream = StringIO(input)
        with stream:
            res = cls.from_mol2(
                stream,
                name=name,
                source_units=source_units,
            )

        return res

    @property
    def n_conformers(self):
        return self._coords.shape[0]

    def __iter__(self):
        self._current_mol_index = 0
        return self

    def __next__(self) -> Conformer:
        idx = self._current_mol_index
        if idx in range(self.n_conformers):
            m = self[idx]
            self._current_mol_index += 1
        else:
            raise StopIteration
        return m

    def __getitem__(self, locator: int | slice) -> Conformer | List[Conformer]:
        match locator:
            case int() as _i:
                return Conformer(self, _i)

            case slice() as _s:
                return [
                    Conformer(self, _i) for _i in range(*_s.indices(self.n_conformers))
                ]

            case _:
                raise ValueError("Cannot use this locator")

    def __str__(self):
        _fml = self.formula if self.n_atoms > 0 else "[no atoms]"
        s = f"ConformerEnsemble(name='{self.name}', formula='{_fml}', n_conformers={self.n_conformers})"
        return s

    def extend(self, others: Iterable[StructureLike]):
        # TODO: Convince Lena to commit these changes
        ...

    def append(self, other: StructureLike):
        # TODO: Convince Lena to commit these changes
        ...

    def filter(
        self,
        fx: Callable[[Conformer], bool],
    ):
        ...

    def serialize(self):
        atom_id_map = {a: i for i, a in enumerate(self.atoms)}

        # atoms = [
        #     (a.element.z, a.label, a.isotope, a.dummy, a.stereo) for a in self.atoms
        # ]
        # bonds = [
        #     (atom_index[b.a1], atom_index[b.a2], b.order, b.aromatic, b.stereo)
        #     for b in self.bonds
        # ]

        atoms = [a.as_tuple() for a in self.atoms]
        bonds = [b.as_tuple(atom_id_map=atom_id_map) for b in self.bonds]

        return (
            self.name,
            self.n_conformers,
            self.n_atoms,
            atoms,
            bonds,
            self.charge,
            self.mult,
            self.coords.astype(">f4").tobytes(),
            self.weights.astype(">f4").tobytes(),
            self.atomic_charges.astype(">f4").tobytes(),
        )

    @classmethod
    def deserialize(cls: type[ConformerEnsemble], struct: tuple):
        (
            _name,
            _nc,
            _na,
            _atoms,
            _bonds,
            _charge,
            _mult,
            _coords,
            _weights,
            _atomic_charges,
        ) = struct

        coords = np.frombuffer(_coords, dtype=">f4").reshape((_nc, _na, 3))
        weights = np.frombuffer(_weights, dtype=">f4")
        atomic_charges = np.frombuffer(_atomic_charges, dtype=">f4")

        atoms = []
        for i, a in enumerate(_atoms):
            atoms.append(Atom(*a))

        res = cls(
            atoms,
            n_conformers=_nc,
            n_atoms=_na,
            name=_name,
            charge=_charge,
            mult=_mult,
            coords=coords,
            weights=weights,
            atomic_charges=atomic_charges,
        )

        for i, b in enumerate(_bonds):
            a1, a2, *b_attr = b
            res.append_bond(Bond(atoms[a1], atoms[a2], *b_attr))

        return res

    def scale(self, factor: float, allow_inversion=False):
        """
        Simple multiplication of all coordinates by a factor.
        Useful for unit conversion.
        """
        if factor < 0 and not allow_inversion:
            raise ValueError(
                "Scaling with a negative factor can only be performed with explicit `scale(factor, allow_inversion = True)`"
            )

        if factor == 0:
            raise ValueError("Scaling with a factor == 0 is not allowed.")

        self._coords *= factor

    def invert(self):
        """
        Coordinates are inverted wrt the origin. This also inverts the absolute stereochemistry
        """
        self.scale(-1, allow_inversion=True)


class Conformer(Molecule):
    """
    Conformer class is a virtual instance that behaves like a molecule,
    yet is completely virtual.
    """

    def __init__(self, parent: ConformerEnsemble, conf_id: int):
        self._parent = parent
        self._conf_id = conf_id

    @property
    def name(self):
        return self._parent.name

    @property
    def _atoms(self):
        return self._parent.atoms

    @property
    def _bonds(self):
        return self._parent.bonds

    @property
    def _coords(self):
        return self._parent._coords[self._conf_id]

    @_coords.setter
    def _coords(self, other):
        self._parent._coords[self._conf_id] = other

    @property
    def charge(self):
        return self._parent.charge

    @property
    def mult(self):
        return self._parent.mult

    def __str__(self):
        _fml = self.formula if self.n_atoms > 0 else "[no atoms]"
        s = f"Conformer(name='{self.name}', conf_id={self._conf_id})"
        return s
