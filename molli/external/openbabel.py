from __future__ import annotations
from typing import Any, List, Iterable, Generator, TypeVar, Generic, Dict
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from io import StringIO, BytesIO
import re
from warnings import warn

from openbabel import openbabel as ob

# def to_obmol(self):
#     obm = ob.OBMol()

#     for i, a in enumerate(self.atoms):
#         oba: ob.OBAtom = obm.NewAtom()
#         oba.SetAtomicNum(a.Z)
#         x, y, z = map(float, self.coords[i])
#         oba.SetVector(x, y, z)

#     for j, b in enumerate(self.bonds):
#         a1, a2 = self.yield_atom_indices((b.a1, b.a2))
#         obb: ob.OBBond = obm.AddBond(a1 + 1, a2 + 1, int(b.order))

#     obm.PerceiveBondOrders()

#     return obm

# @classmethod
# def from_obmol(cls: type[Molecule], other: ob.OBMol):
#     n_atoms = other.NumAtoms()
#     n_bonds = other.NumBonds()
#     charge = other.GetTotalCharge()
#     mult = other.GetTotalSpinMultiplicity()
#     mol = Molecule(n_atoms, charge=charge, multiplicity=mult)

#     for i in range(n_atoms):
#         obat: ob.OBAtom = other.GetAtomById(i)
#         mol.atoms[i].element = Element.get(obat.GetAtomicNum())
#         mol.coords[i] = [obat.GetX(), obat.GetY(), obat.GetZ()]

#     for j in range(n_bonds):
#         obbd: ob.OBBond = other.GetBondById(j)
#         i1 = obbd.GetBeginAtomIdx()
#         i2 = obbd.GetEndAtomIdx()
#         order = obbd.GetBondOrder()
#         mol.new_bond(i1 - 1, i2 - 1, order)

#     return mol


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
