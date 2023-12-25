# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by  Blake E. Ocampo
#               Alexander S. Shved
#
# S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign
# https://denmarkgroup.illinois.edu/
#
# Copyright 2022-2023 The Board of Trustees of the University of Illinois.
# All Rights Reserved.
#
# Licensed under the terms MIT License
# The License is included in the distribution as LICENSE file.
# You may not use this file except in compliance with the License.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# ================================================================================


"""
This module provides a set of functions (albeit incomplete) to interface with
OpenBabel package
"""

from __future__ import annotations
from ..chem import Molecule, ConformerEnsemble, Element, Bond
from typing import Any, List, Iterable, Generator, TypeVar, Generic, Dict
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike
from io import StringIO, BytesIO
import re
from warnings import warn
from tempfile import NamedTemporaryFile
import molli as ml
import os
from ctypes import POINTER, c_double

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
        oba.SetAtomicNum(
            a.Z if a.element is not Element.Unknown else Element.get(dummy).z
        )
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


def from_obmol(obmol: ob.OBMol, cls: type = Molecule) -> Molecule:
    """
    This is only a stub method right now. We need fully supported conversion.
    """
    n_atoms = obmol.NumAtoms()
    n_bonds = obmol.NumBonds()
    obmol.NumConformers

    charge = obmol.GetTotalCharge()
    mult = obmol.GetTotalSpinMultiplicity()
    name = obmol.GetTitle()
    mol = cls(name=name, n_atoms=n_atoms, charge=charge, mult=mult)
    for i in range(n_atoms):
        a: ob.OBAtom = obmol.GetAtomById(i)
        mol.coords[i] = [a.GetX(), a.GetY(), a.GetZ()]
        mol.atoms[i].element = a.GetAtomicNum()
        mol.atoms[i].isotope = a.GetIsotope()

    for j in range(n_bonds):
        b: ob.OBBond = obmol.GetBondById(j)
        ai1, ai2 = b.GetBeginAtomIdx() - 1, b.GetEndAtomIdx() - 1
        mol.connect(ai1, ai2)

    return mol


def loads_obmol(
    inp: str, ext: str = "xyz", connect_perceive: bool = False, cls: type = Molecule
) -> ob.OBMol:
    """
    This function takes any file and creates an openbabel style mol format
    """

    conv = ob.OBConversion()
    obmol = ob.OBMol()
    conv.SetInFormat(ext)
    conv.ReadString(obmol, inp)

    if connect_perceive:
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()

    mlmol = from_obmol(obmol=obmol, cls=cls)

    return mlmol


def loads_all_obmol(data, ext, connect_perceive: bool = False, cls: type = Molecule):
    """
    Requires creation of a temporary file, loads all files in an object
    """
    with NamedTemporaryFile(
        dir=ml.config.SCRATCH_DIR,
        prefix="obabel-",
        suffix=f"{ext}",
        mode="w+b",
        delete=False,
    ) as tf:
        tf.write(data)

    conv = ob.OBConversion()
    conv.SetInFormat(ext)
    conv.SetOptions("m", ob.OBConversion.INOPTIONS)
    obmol = ob.OBMol()
    notatend = conv.ReadFile(obmol, tf.name)
    mols = list()

    while notatend:
        mlmol = from_obmol(obmol)
        if connect_perceive:
            obmol.ConnectTheDots()
            obmol.PerceiveBondOrders()
        mols.append(mlmol)
        obmol = ob.OBMol()
        notatend = conv.Read(obmol)

    os.remove(tf.name)

    return mols


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


def dumps_obmol(mol: Molecule | ConformerEnsemble, ftype: str, encode=False):
    """
    This returns basic file data when writing the file that would be expected using Avogadro/Openbabel
    """
    if isinstance(mol, Molecule):
        obm = to_obmol(mol)
        conv = ob.OBConversion()
        conv.SetOutFormat(ftype)
        if encode:
            return conv.WriteString(obm).encode()
        else:
            return conv.WriteString(obm)

    elif isinstance(mol, ConformerEnsemble):
        write = list()
        for conf in mol:
            obm = to_obmol(conf)
            conv = ob.OBConversion()
            conv.SetOutFormat(ftype)
            write.append(conv.WriteString(obm))
        write = "".join(write)
        if encode:
            return write.encode()
        else:
            return write


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
