from __future__ import annotations
from typing import Any, List, Iterable, Generator, TypeVar, Generic, Dict
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from io import StringIO, BytesIO
import re
from warnings import warn
from openbabel import openbabel as ob

from . import (
    Element,
    Atom,
    AtomLike,
    Bond,
    Connectivity,
    CartesianGeometry,
    Structure,
    Substructure,
)


class Molecule(Structure):
    """Fundamental class of the MOLLI Package."""

    __slots__ = ("_charge", "multiplicity", "_partial_charges") + Structure.__slots__

    def __init__(
        self,
        n_atoms: int = 0,
        charge: int = 0,
        multiplicity: int = 1,
        name: str = "unnamed",
        *,
        atomic_charges: ArrayLike = ...,
        dtype: int = "float32",
    ):
        """
        If other is not none, that molecule will be cloned.
        """
        # if isinstance(other, Molecule | Structure):
        #     ...
        super().__init__(n_atoms=n_atoms, name=name, dtype=dtype)
        self.charge = charge
        self.multiplicity = multiplicity
        self.atomic_charges = atomic_charges

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, other: int):
        self._charge = other

    @property
    def atomic_charges(self):
        return self._atomic_charges

    @atomic_charges.setter
    def atomic_charges(self, other: ArrayLike):
        if other is Ellipsis:
            self._atomic_charges = np.zeros(self.n_atoms, dtype=self._dtype)
        else:
            _pc = np.array(other, dtype=self._dtype)
            if _pc.shape == (self.n_atoms,):
                self._atomic_charges = _pc
            else:
                raise ValueError("Inappropriate shape of atomic charge array")

    def __str__(self):
        _fml = self.formula if self.n_atoms > 0 else "[no atoms]"
        s = f"Molecule(name='{self.name}', formula='{_fml}')"
        return s
        ...

    def dump_mol2(self, stream: StringIO):
        stream.write(f"# Produced with molli package\n")
        stream.write(
            f"@<TRIPOS>MOLECULE\n{self.name}\n{self.n_atoms} {self.n_bonds} 0 0 0\nSMALL\nUSER_CHARGES\n\n"
        )

        stream.write("@<TRIPOS>ATOM\n")
        for i, a in enumerate(self.atoms):
            x, y, z = self.coords[i]
            c = self.atomic_charges[i]
            stream.write(
                f"{i+1:>6} {a.label:<3} {x:>12.6f} {y:>12.6f} {z:>12.6f} {a.element.symbol:<10} 1 UNL1 {c}\n"
            )

        stream.write("@<TRIPOS>BOND\n")
        for i, b in enumerate(self.bonds):
            a1, a2 = self.atoms.index(b.a1), self.atoms.index(b.a2)
            bond_type = "ar" if b.aromatic else f"{b.order:1.0f}"
            stream.write(f"{i+1:>6} {a1+1:>6} {a2+1:>6} {bond_type:>10}\n")

    def to_obmol(self):
        obm = ob.OBMol()

        for i, a in enumerate(self.atoms):
            oba: ob.OBAtom = obm.NewAtom()
            oba.SetAtomicNum(a.Z)
            x, y, z = map(float, self.coords[i])
            oba.SetVector(x, y, z)

        for j, b in enumerate(self.bonds):
            a1, a2 = self.yield_atom_indices((b.a1, b.a2))
            obb: ob.OBBond = obm.AddBond(a1 + 1, a2 + 1, int(b.order))
        
        obm.PerceiveBondOrders()

        return obm

    @classmethod
    def from_obmol(cls: type[Molecule], other: ob.OBMol):
        n_atoms = other.NumAtoms()
        n_bonds = other.NumBonds()
        charge = other.GetTotalCharge()
        mult = other.GetTotalSpinMultiplicity()
        mol = Molecule(n_atoms, charge=charge, multiplicity=mult)

        for i in range(n_atoms):
            obat: ob.OBAtom = other.GetAtomById(i)
            mol.atoms[i].element = Element.get(obat.GetAtomicNum())
            mol.coords[i] = [obat.GetX(), obat.GetY(), obat.GetZ()]

        for j in range(n_bonds):
            obbd: ob.OBBond = other.GetBondById(j)
            i1 = obbd.GetBeginAtomIdx()
            i2 = obbd.GetEndAtomIdx()
            order = obbd.GetBondOrder()
            mol.new_bond(i1 - 1, i2 - 1, order)

        return mol


def obabel_load(fname: str, input_ext: str = "mol2"):
    conv = ob.OBConversion()
    obmol = ob.OBMol()
    conv.SetInFormat(input_ext)
    conv.ReadFile(obmol, fname)

    n_atoms = obmol.NumAtoms()
    n_bonds = obmol.NumBonds()
    mlmol = Molecule([Element.Unknown for _ in range(n_atoms)])

    for i in range(n_atoms):
        obat: ob.OBAtom = obmol.GetAtomById(i)
        mlmol.atoms[i].element = Element.get(obat.GetAtomicNum())
        mlmol.coords[i] = [obat.GetX(), obat.GetY(), obat.GetZ()]

    for j in range(n_bonds):
        obbd: ob.OBBond = obmol.GetBondById(j)
        i1 = obbd.GetBeginAtomIdx()
        i2 = obbd.GetEndAtomIdx()
        order = obbd.GetBondOrder()
        mlmol.new_bond(i1 - 1, i2 - 1, order)

    return mlmol


StructureLike = Molecule | Structure | Substructure
