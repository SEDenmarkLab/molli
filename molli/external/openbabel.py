from __future__ import annotations
from ..chem import Molecule, Element, Bond
from typing import Any, List, Iterable, Generator, TypeVar, Generic, Dict
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from io import StringIO, BytesIO
import re
from warnings import warn

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

try:
    from openbabel import openbabel as ob
except:
    raise ImportError("OpenBabel is not installed in this environment")


# def obabel_load(fname: str, input_ext: str = "mol2"):
#     conv = ob.OBConversion()
#     obmol = ob.OBMol()
#     conv.SetInFormat(input_ext)
#     conv.ReadFile(obmol, fname)

#     n_atoms = obmol.NumAtoms()
#     n_bonds = obmol.NumBonds()
#     mlmol = Molecule([Element.Unknown for _ in range(n_atoms)])

#     for i in range(n_atoms):
#         obat: ob.OBAtom = obmol.GetAtomById(i)
#         mlmol.atoms[i].element = Element.get(obat.GetAtomicNum())
#         mlmol.coords[i] = np.array([obat.GetX(), obat.GetY(), obat.GetZ()])

#     for j in range(n_bonds):
#         obbd: ob.OBBond = obmol.GetBondById(j)
#         i1 = obbd.GetBeginAtomIdx()
#         i2 = obbd.GetEndAtomIdx()
#         order = obbd.GetBondOrder()
#         mlmol.bonds[j] = Bond(a1 = i1 - 1, a2= i2 - 1, btype=order)

#     return mlmol

def to_mol2_w_ob(molli_mol: Molecule):
    '''
    This returns basic mol2 data when writing the mol2 file that would be expected using Avogadro/Openbabel
    '''
    obm = ob.OBMol()

    for i, a in enumerate(molli_mol.atoms):
        oba: ob.OBAtom = obm.NewAtom()
        oba.SetAtomicNum(a.Z)
        x, y, z = map(float, molli_mol.coords[i])
        oba.SetVector(x, y, z)

    for j, b in enumerate(molli_mol.bonds):
        a1_idx = molli_mol.get_atom_index(b.a1)
        a2_idx = molli_mol.get_atom_index(b.a2)
        obb: ob.OBBond = obm.AddBond(a1_idx + 1, a2_idx + 1, int(b.order))
    
    obm.PerceiveBondOrders()
    obm.SetTitle(molli_mol.name)
    conv = ob.OBConversion()
    conv.SetOutFormat('mol2')

    return conv.WriteString(obm)

def to_file_w_ob(molli_mol: Molecule,ftype:str):
    '''
    This returns basic file data when writing the file that would be expected using Avogadro/Openbabel
    '''
    obm = ob.OBMol()

    for i, a in enumerate(molli_mol.atoms):
        oba: ob.OBAtom = obm.NewAtom()
        oba.SetAtomicNum(a.Z)
        x, y, z = map(float, molli_mol.coords[i])
        oba.SetVector(x, y, z)

    for j, b in enumerate(molli_mol.bonds):
        a1_idx = molli_mol.get_atom_index(b.a1)
        a2_idx = molli_mol.get_atom_index(b.a2)
        obb: ob.OBBond = obm.AddBond(a1_idx + 1, a2_idx + 1, int(b.order))
    
    obm.PerceiveBondOrders()
    obm.SetTitle(molli_mol.name)
    conv = ob.OBConversion()
    conv.SetOutFormat(ftype)

    return conv.WriteString(obm)

def opt_w_ob(mlmol:Molecule, ff = 'UFF', inplace = False) -> Molecule:
    '''
    If inplace = True, this mutates mlmol to optimized coordinates and returns None
    If inplace = False, mlmol is unchanged and an optimized copy is returned
    '''
    warn('This function may not be the final version. Current iteration only contains atoms and bonds, not charges and multiplicity.')
    obm = ob.OBMol()

    for i, a in enumerate(mlmol.atoms):
        oba: ob.OBAtom = obm.NewAtom()
        oba.SetAtomicNum(a.Z)
        x, y, z = map(float, mlmol.coords[i])
        oba.SetVector(x, y, z)

    for j, b in enumerate(mlmol.bonds):
        a1_idx = mlmol.get_atom_index(b.a1)
        a2_idx = mlmol.get_atom_index(b.a2)
        obb: ob.OBBond = obm.AddBond(a1_idx + 1, a2_idx + 1, int(b.order))

    obf = ob.OBForceField.FindForceField(ff)
    obf.Setup(obm)
    obf.ConjugateGradients(500)
    obf.GetCoordinates(obm)

    if inplace == True:
        for i, a in enumerate(mlmol.atoms):
            ob_atom = obm.GetAtomById(i)
            new_coords = np.array((ob_atom.GetX(),ob_atom.GetY(), ob_atom.GetZ()))
            mlmol.coords[i] = new_coords
        return None
    else:
        rtn = Molecule(mlmol)
        for i, a in enumerate(rtn.atoms):
            ob_atom = obm.GetAtomById(i)
            new_coords = np.array((ob_atom.GetX(),ob_atom.GetY(), ob_atom.GetZ()))
            rtn.coords[i] = new_coords
        return rtn