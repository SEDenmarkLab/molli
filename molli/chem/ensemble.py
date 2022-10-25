from __future__ import annotations
from typing import Iterable, Iterator, List
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
from .geometry import _nans
from .structure import RE_MOL_NAME, RE_MOL_ILLEGAL
from ..parsing import read_mol2

import numpy as np
from numpy.typing import ArrayLike
from warnings import warn
from io import StringIO


class ConformerEnsemble(Connectivity):
    def __init__(
        self,
        n_conformers: int = 0,
        n_atoms: int = 0,
        *,
        name: str = "unnamed",
        charge: int = 0,
        multiplicity: int = 1,
        weights: ArrayLike = ...,
        atomic_charges: ArrayLike = ...,
        dtype: str = "float32",
    ):
        self.charge = charge
        self.multiplicity = multiplicity
        self._dtype = dtype

        super().__init__(n_atoms, name=name)

        self.weights = weights
        self.atomic_charges = atomic_charges
        self._coords = _nans((n_conformers, n_atoms, 3), dtype)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, other: ArrayLike):
        if other is Ellipsis:
            self._weights = Ellipsis
        else:
            _weights = np.array(other, dtype=self._dtype)
            if _weights.shape == (self.n_conformers,):
                self._weights = _weights
            else:
                raise ValueError(f"Inappropriate shape for conformer weights")

    @property
    def atomic_charges(self):
        return self._atomic_charges

    @atomic_charges.setter
    def atomic_charges(self, other: ArrayLike):
        if other is Ellipsis:
            self._atomic_charges = Ellipsis
        else:
            _charges = np.array(other, dtype=self.dtype)
            if _charges.shape == (self.n_atoms,) or _charges.shape == (
                self.n_conformers,
                self.n_atoms,
            ):
                self._atomic_charges = _charges
            else:
                raise ValueError(f"Inappropriate shape for atomic charges")

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
                return [Conformer(self, _i) for _i in range(_s.start, _s.stop, _s.step)]

            case _:
                raise ValueError("Cannot use this locator")

    def __str__(self):
        _fml = self.formula if self.n_atoms > 0 else "[no atoms]"
        s = f"ConformerEnsemble(name='{self.name}', formula='{_fml}', n_conformers={self.n_conformers})"
        return s

    def extend(self, others: Iterable[Molecule]):
        ...

    def append(self, other: Iterable[Molecule]):
        ...

    @classmethod
    def from_mol2(
        cls: type[ConformerEnsemble],
        input: str | StringIO,
        /,
        name: str = ...,
        charge: int = 0,
        multiplicity: int = 1,
        dtype: str = "float32",
    ) -> ConformerEnsemble:
        """ """
        mol2io = StringIO(input) if isinstance(input, str) else input
        mols = list(Molecule.yield_from_mol2(mol2io, name=name, dtype=dtype))

        res = cls(
            mols[0],
            len(mols),
            charge=charge,
            multiplicity=multiplicity,
            dtype=dtype,
        )

        for i, m in enumerate(mols):
            res._coords[i] = m.coords

        return res

    def serialize(self):
        atom_index = {a: i for i, a in enumerate(self.atoms)}
        atoms = [
            (a.element.z, a.label, a.isotope, a.dummy, a.stereo) 
            for a in self.atoms
        ]
        bonds = [
            (atom_index[b.a1], atom_index[b.a2], b.order, b.aromatic, b.stereo)
            for b in self.bonds
        ]

        return (
            self.name,
            self.n_conformers,
            self.n_atoms,
            atoms,
            bonds,
            self.charge,
            self.multiplicity,
            self._dtype,
            None if self._coords is Ellipsis else self._coords.astype(">f4").tobytes(),
            None if self.weights is Ellipsis else self.weights.astype(">f4").tobytes(),
            None
            if self.atomic_charges is Ellipsis
            else self.atomic_charges.astype(">f4").tobytes(),
        )

    @classmethod
    def deserialize(cls: type[ConformerEnsemble], struct: tuple):
        (
            name,
            nc,
            na,
            atoms,
            bonds,
            charge,
            mult,
            dt,
            _coords,
            _weights,
            _atomic_charges,
        ) = struct

        coords = None if _coords is None else np.frombuffer(_coords, dtype=">f4")
        weights = None if _weights is None else np.frombuffer(_weights, dtype=">f4")
        atomic_charges = (
            None
            if _atomic_charges is None
            else np.frombuffer(_atomic_charges, dtype=">f4")
        )

        res = cls(
            nc,
            na,
            name=name,
            charge=charge,
            multiplicity=mult,
            dtype=dt,
        )

        atom_index = {}

        for i, a in enumerate(atoms):
            elt, lbl, iso, dum, ste = a
            a = res.atoms[i]
            a.element = Element.get(elt)
            a.label = lbl
            a.isotope = iso
            a.dummy = (dum,)
            a.stereo = ste
            atom_index[i] = a

        for i, b in enumerate(bonds):
            a1, a2, order, arm, ste = b
            Bond.add_to(
                res,
                atom_index[a1],
                atom_index[a2],
                order=order,
                stereo=ste,
                aromatic=arm,
            )

        res._coords = np.array(coords, dtype=res._dtype).reshape((nc, na, 3))
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

    def __init__(self, parent: ConformerEnsemble, cid: int):
        self._parent = parent
        self._cid = cid

    @property
    def name(self):
        return self._parent.name

    @property
    def _atoms(self):
        return self._parent.atoms

    @property
    def _dtype(self):
        return self._parent._dtype

    @property
    def _bonds(self):
        return self._parent.bonds

    @property
    def _coords(self):
        return self._parent._coords[self._cid]

    @_coords.setter
    def _coords(self, other):
        self._parent._coords[self._cid] = other

    @property
    def charge(self):
        return self._parent.charge

    @property
    def multiplicity(self):
        return self._parent.multiplicity

    def __str__(self):
        _fml = self.formula if self.n_atoms > 0 else "[no atoms]"
        s = f"Conformer(name='{self.name}', cid={self._cid})"
        return s
