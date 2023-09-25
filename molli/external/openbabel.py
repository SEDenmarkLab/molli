from __future__ import annotations
from ..chem import Molecule, Element, Bond
from typing import Any, List, Iterable, Generator, TypeVar, Generic, Dict
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from io import StringIO, BytesIO
import re
from warnings import warn

try:
    from openbabel import openbabel as ob
except:
    raise ImportError("OpenBabel is not installed in this environment")


def to_obmol(
    mol: Molecule,
    *,
    coord_displace: float | bool = False,
    dummy: Element | str = Element.Cl,
) -> ob.OBMol:
    """
    This function takes a molli Molecule object and generates an openbabel molecule.
    dummy parameter: replace the dummy atoms with this element on the fly
    coord_displace: add a random displacement vector of this magnitude
    """
    obm = ob.OBMol()

    obm.BeginModify()

    for i, a in enumerate(mol.atoms):
        oba: ob.OBAtom = obm.NewAtom()
        oba.SetAtomicNum(a.Z if a.element is not Element.Unknown else Element.get(dummy).z)
        if coord_displace:
            dvec = np.random.random(3)
            dvec *= coord_displace / np.linalg.norm(dvec)
            x, y, z = mol.coords[i] + dvec
        else:
            x, y, z = mol.coords[i]
        oba.SetVector(x, y, z)
        oba.SetFormalCharge(0)

    for j, b in enumerate(mol.bonds):
        a1_idx = mol.get_atom_index(b.a1)
        a2_idx = mol.get_atom_index(b.a2)
        obb: ob.OBBond = obm.AddBond(a1_idx + 1, a2_idx + 1, int(b.order))

    obm.SetTotalCharge(mol.charge)
    obm.SetTotalSpinMultiplicity(mol.mult)
    obm.EndModify()
    obm.PerceiveBondOrders()

    obm.SetTitle(mol.name)

    return obm


def coord_from_obmol(obmol: ob.OBMol, dtype: np.dtype = np.float64) -> np.ndarray:
    n_atoms = obmol.NumAtoms()
    coord = np.empty((n_atoms, 3), dtype=dtype)
    for i in range(n_atoms):
        a = obmol.GetAtomById(i)
        coord[i] = [a.GetX(), a.GetY(), a.GetZ()]

    return coord


def coords_to_obmol(obmol: ob.OBMol, coords: np.ndarray):
    n_atoms = obmol.NumAtoms()
    for i in range(n_atoms):
        a = obmol.GetAtomById(i)
        x, y, z = map(float, coords[i])
        a.SetVector(x, y, z)


def from_str_w_ob(block: str, input_fmt: str = "mol2") -> ob.OBMol:
    """
    This function takes any file and creates an openbabel style mol format
    """
    
    conv = ob.OBConversion()
    obmol = ob.OBMol()
    conv.SetInFormat(input_fmt)
    conv.ReadString(obmol, block)
    obmol.PerceiveBondOrders()
    return obmol


def to_mol2_w_ob(mol: Molecule):
    """
    This returns basic mol2 data when writing the mol2 file that would be expected using Avogadro/Openbabel
    """
    obm = to_obmol(mol)
    conv = ob.OBConversion()
    conv.SetOutFormat("mol2")

    return conv.WriteString(obm)


def to_file_w_ob(mol: Molecule, ftype: str):
    """
    This returns basic file data when writing the file that would be expected using Avogadro/Openbabel
    """
    obm = to_obmol(mol)
    conv = ob.OBConversion()
    conv.SetOutFormat(ftype)

    return conv.WriteString(obm)


def obabel_optimize(
    mol: Molecule,
    *,
    ff: str = "UFF",
    max_steps: int = 1000,
    coord_displace: float | bool = False,
    tol: float = 1e-4,
    dummy: Element | str = Element.Cl,
    inplace: bool = False,
) -> Molecule:
    """
    If inplace = True, this mutates mlmol to optimized coordinates and returns None
    If inplace = False, mlmol is unchanged and an optimized copy is returned
    """

    obm = to_obmol(mol, dummy=dummy, coord_displace=coord_displace)
    obff = ob.OBForceField.FindForceField(ff)

    obff.Setup(obm)
    obff.SteepestDescent(max_steps, tol)
    obff.GetCoordinates(obm)

    optimized = coord_from_obmol(obm)

    if inplace:
        mol.coords = optimized
    else:
        return Molecule(mol, coords=optimized)
