# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Blake E. Ocampo
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
# `molli.external._rdkit`
This module defines necessary functions (albeit not a complete set) to interface with RDKit.
"""

from ..chem import Molecule
from typing import Dict

import numpy as np
import importlib.util


def is_package_installed(pkg_name):
    return importlib.util.find_spec(pkg_name) is not None


if not is_package_installed("rdkit"):
    raise ImportError("RDKit is not installed in this environment")

if not is_package_installed("IPython"):
    raise ImportError("IPython is not installed in this environment")

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.PropertyMol import PropertyMol
from rdkit.Chem import rdqueries as chemq
from rdkit import DataStructs
from rdkit.Chem import rdCIPLabeler
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole


def visualize_mols(
    name: str,
    rdkit_mol_list: list,
    molsPerRow=5,
    prop: str = "_Name",
    svg=True,
    png=False,
):
    """
    This visualize any RDKit mols that can be kekulized
    """

    if svg:
        _img = Draw.MolsToGridImage(
            rdkit_mol_list,
            molsPerRow=molsPerRow,
            subImgSize=(200, 200),
            useSVG=True,
            returnPNG=False,
            legends=[i.GetProp(prop) for i in rdkit_mol_list],
            maxMols=1000,
        )
        with open(f"{name}.svg", "w") as f:
            f.write(_img.data)
    if png:
        _img = Draw.MolsToGridImage(
            rdkit_mol_list,
            molsPerRow=molsPerRow,
            subImgSize=(200, 200),
            useSVG=False,
            returnPNG=True,
            legends=[i.GetProp(prop) for i in rdkit_mol_list],
            maxMols=1000,
        )
        with open(f"{name}.png", "wb") as f:
            f.write(_img.data)


def create_rdkit_mol(
    molli_mol: Molecule, removeHs=False
) -> Dict[Molecule, PropertyMol]:
    """
    Uses mol2 generated from openbabel's implementation of mol2 generation.
    """
    from .openbabel import to_mol2_w_ob

    try:
        rdkit_mol = PropertyMol(
            Chem.MolFromMol2Block(to_mol2_w_ob(molli_mol), removeHs=removeHs)
        )
        rdkit_mol.SetProp("_Name", f"{molli_mol.name}")
    except:
        rdkit_mol = PropertyMol(
            Chem.MolFromMol2Block(
                to_mol2_w_ob(molli_mol), removeHs=removeHs, sanitize=False
            )
        )
        rdkit_mol.SetProp("_Name", f"{molli_mol.name}")
        rdkit_mol.SetProp("_Kekulize_Issue", "1")

    return {molli_mol: rdkit_mol}


def visualize_molli_mol(
    name: str, molli_mol_list: list, removeHs=True, molsPerRow=5, svg=True, png=False
):
    """
    This does a basic 2D visualization of the molli_mols
    """

    final_visual = list()
    no_visual = list()
    for mlmol in molli_mol_list:
        _entry = create_rdkit_mol(mlmol, removeHs=removeHs)
        if _entry[mlmol].HasProp("_Kekulize_Issue"):
            no_visual.append(_entry[mlmol].GetProp("_Name"))
            continue
        else:
            final_visual.append(_entry[mlmol])

    if len(final_visual) == 0:
        print("No molecules successfully kekulized for visualization")
        return None

    if len(no_visual) != 0:
        print(f"\nUnable to visualize the following molecules:\n{no_visual}")

    if svg:
        _img = Draw.MolsToGridImage(
            final_visual,
            molsPerRow=molsPerRow,
            subImgSize=(200, 200),
            useSVG=True,
            returnPNG=False,
            legends=[i.GetProp("_Name") for i in final_visual],
            maxMols=1000,
        )
        with open(f"{name}.svg", "w") as f:
            f.write(_img.data)
    if png:
        _img = Draw.MolsToGridImage(
            final_visual,
            molsPerRow=molsPerRow,
            subImgSize=(200, 200),
            useSVG=False,
            returnPNG=True,
            legends=[i.GetProp("_Name") for i in final_visual],
            maxMols=1000,
        )
        with open(f"{name}.png", "wb") as f:
            f.write(_img.data)


def canonicalize_rdkit_mol(rdkit_mol, sanitize=False) -> PropertyMol:
    """
    Returns canonicalized RDKit mol generated from a canonicalized RDKit SMILES string.

    Can indicate whether hydrogens should be removed from mol object or not.

    Sanitizing will also remove hydrogens

    The Current implementation will only add the "_Name" property
    """

    can_smiles = Chem.MolToSmiles(rdkit_mol, canonical=True)
    can_rdkit_mol = Chem.MolFromSmiles(can_smiles, sanitize=sanitize)

    if rdkit_mol.HasProp("_Name"):
        can_rdkit_mol.SetProp("_Name", rdkit_mol.GetProp("_Name"))

    return can_rdkit_mol


def can_mol_order(rdkit_mol):
    """
    This a function tries to match the indexes of the canonicalized smiles string/molecular graph to a Molli Molecule object.
    Any inputs to this function will AUTOMATICALLY ADD HYDROGENS (make them explicit) to the RDKit mol object. This function returns 3 objects:

    1. Canonical RDKit Mol Object with Hydrogens and all maintained properties from the original rdkit mol
    2. A List for reordering the Atom Indices after canonicalization
    3. A list for reordering the Bond Indices after canonicalization

    Important Notes:
    - It will only have "_Kekulize_Issue" if the initial object had this property set (i.e. if it ran into an issue in the in initial instantiation)
    - The canonical rdkit mol object will have the "Canonical SMILES with hydrogens" available as the property: "_Canonical_SMILES_w_H"
    - There may be some properties missing as the PropertyCache is not being updated on the new canonicalized mol object, so consider using rdkit_mol.UpdatePropertyCache() if you want to continue using the mol object
    """

    # This is here to deal with any smiles strings or mol objects that do not get assigned hydrogens
    new_rdkit_mol = Chem.AddHs(rdkit_mol)

    #### This statement is necessary to generate the mol.GetPropertyName "_smilesAtomOutputOrder" and"_smilesBondOutputOrder"######
    Chem.MolToSmiles(new_rdkit_mol, canonical=True)

    # The smiles output order is actually a string of the form "[0,1,2,3,...,12,]", so it requires a start at 1 and end at -2!
    can_atom_reorder = list(
        map(int, new_rdkit_mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(","))
    )
    canonical_bond_reorder_list = list(
        map(int, new_rdkit_mol.GetProp("_smilesBondOutputOrder")[1:-2].split(","))
    )
    can_smiles_w_h = Chem.MolToSmiles(new_rdkit_mol, canonical=True)

    # # # #Allows maintaining of hydrogens when Mol object is created
    can_mol_w_h = PropertyMol(Chem.MolFromSmiles(can_smiles_w_h, sanitize=False))
    # Certain odd molecules result in some odd calculated properties, so this part is remaining commented out for now
    # can_mol_w_h.UpdatePropertyCache()
    all_props_original_rdkit_mol = list(rdkit_mol.GetPropNames())

    # Helps new rdkit object maintain original properties of rdkit mol put in
    for prop in all_props_original_rdkit_mol:
        if not can_mol_w_h.HasProp(prop):
            can_mol_w_h.SetProp(prop, rdkit_mol.GetProp(prop))

    can_mol_w_h.SetProp("_Canonical_SMILES_w_H", f"{can_smiles_w_h}")

    return can_mol_w_h, can_atom_reorder, canonical_bond_reorder_list


def reorder_molecule(
    molli_mol: Molecule,
    can_rdkit_mol_w_h,
    can_atom_reorder: list,
    can_bond_reorder: list,
):
    """
    This is a function that utilizes the outputs of new_mol_order to reorder an existing molecule.
    Currently done in place on the original molli_mol object.
    """

    # This reorders the atoms of the molecule object
    molli_atoms_arr = np.array(molli_mol.atoms)
    fixed_atom_order_list = molli_atoms_arr[can_atom_reorder].tolist()
    molli_mol._atoms = fixed_atom_order_list

    # This reorders the bonds of the molecule object
    molli_obj_bonds_arr = np.array(molli_mol.bonds)
    fixed_bond_order_list = molli_obj_bonds_arr[can_bond_reorder].tolist()
    molli_mol._bonds = fixed_bond_order_list

    # This fixes the geometry of the molecule object
    molli_mol.coords = molli_mol.coords[can_atom_reorder]

    # This checks to see if the new rdkit atom order in the canonical smiles matches the new molli order of atoms
    can_rdkit_atoms = can_rdkit_mol_w_h.GetAtoms()
    can_rdkit_atom_elem = np.array([x.GetSymbol() for x in can_rdkit_atoms])

    new_molli_elem = np.array([atom.element.symbol for atom in molli_mol.atoms])
    equal_check = np.array_equal(can_rdkit_atom_elem, new_molli_elem)

    assert (
        equal_check
    ), f"Array of rdkit atoms: {can_rdkit_atom_elem} is not equal to array of molli atoms: {new_molli_elem}"

    return {molli_mol: can_rdkit_mol_w_h}


class rdkit_atom_filter(Chem.Mol):
    """
    These functions are written as numpy arrays to isolate types of atoms very easily with the goal of the structure being:

    isolated_atoms = (aromatic_type() & sp2_type() & carbon)

    All the functions of this class figure create a boolean array the size of the "All Atoms" array, and then
    they define the intersection of this array with a second array (the case of the condition), and return the array: [(1 & 2 & ...)]

    It is recommended that rdkit molecules are canonicalized before utilizing this function.
    """

    def __init__(self, rdkit_mol):
        self.rdkit_mol = rdkit_mol
        self.atoms = rdkit_mol.GetAtoms()
        self.atoms_array = np.array([x.GetIdx() for x in self.atoms])

        if np.all(np.diff(self.atoms_array) >= 0):
            pass
        else:
            raise ValueError(f"The atom IDs are not ordered from least to greatest")

    def sp2_type(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where SP2 atoms exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        sp2_atoms = chemq.HybridizationEqualsQueryAtom(Chem.HybridizationType.SP2)
        sp2 = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(sp2_atoms)]
        )
        sp2_bool = np.in1d(self.atoms_array, sp2)
        return sp2_bool

    def aromatic_type(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where AROMATIC atoms exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        aromatic_atoms = chemq.IsAromaticQueryAtom()
        aromatic = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(aromatic_atoms)]
        )
        aromatic_bool = np.in1d(self.atoms_array, aromatic)
        return aromatic_bool

    def ring_type(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN A RING exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        ring_atoms = chemq.IsInRingQueryAtom()
        ring = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(ring_atoms)]
        )
        ring_bool = np.in1d(self.atoms_array, ring)
        return ring_bool

    def carbon_type(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where CARBON atoms exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        carbon_atoms = chemq.AtomNumEqualsQueryAtom(6)
        carbon = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(carbon_atoms)]
        )
        carbon_bool = np.in1d(self.atoms_array, carbon)
        return carbon_bool

    def nitrogen_type(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where NITROGEN atoms exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        nitrogen_atoms = chemq.AtomNumEqualsQueryAtom(7)
        nitrogen = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(nitrogen_atoms)]
        )
        nitrogen_bool = np.in1d(self.atoms_array, nitrogen)
        return nitrogen_bool

    def oxygen_type(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where OXYGEN atoms exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        oxygen_atoms = chemq.AtomNumEqualsQueryAtom(8)
        oxygen = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(oxygen_atoms)]
        )
        oxygen_bool = np.in1d(self.atoms_array, oxygen)
        return oxygen_bool

    def atom_num_less_than(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the ATOM NUMBER is LESS than the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        num_light_atoms = chemq.AtomNumLessQueryAtom(number)
        num_light_atom = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(num_light_atoms)]
        )
        num_light_atom_bool = np.in1d(self.atoms_array, num_light_atom)
        return num_light_atom_bool

    def atom_num_equals(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the ATOM NUMBER is EQUAL to the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        num_equals_atoms = chemq.AtomNumEqualsQueryAtom(number)
        num_equal_atom = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(num_equals_atoms)]
        )
        num_equal_atom_bool = np.in1d(self.atoms_array, num_equal_atom)
        return num_equal_atom_bool

    def atom_num_greater_than(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the ATOM NUMBER is GREATER than the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        num_heavy_atoms = chemq.AtomNumGreaterQueryAtom(number)
        num_heavy_atom = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(num_heavy_atoms)]
        )
        num_heavy_atom_bool = np.in1d(self.atoms_array, num_heavy_atom)
        return num_heavy_atom_bool

    def isotope_type_equals(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the ISOTOPE NUMBER is EQUAL to the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        isotope_atoms = chemq.IsotopeEqualsQueryAtom(number)
        isotope_atom = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(isotope_atoms)]
        )
        isotope_atom_bool = np.in1d(self.atoms_array, isotope_atom)
        return isotope_atom_bool

    def charge_type_less_than(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the FORMAL CHARGE is LESS THAN the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        charge_less_atoms = chemq.FormalChargeLessQueryAtom(number)
        charge_less_atom = np.array(
            [
                x.GetIdx()
                for x in self.rdkit_mol.GetAtomsMatchingQuery(charge_less_atoms)
            ]
        )
        charge_less_atom_bool = np.in1d(self.atoms_array, charge_less_atom)
        return charge_less_atom_bool

    def charge_type_equals(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the FORMAL CHARGE is EQUAL to the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        charge_equals_atoms = chemq.FormalChargeEqualsQueryAtom(number)
        charge_equals_atom = np.array(
            [
                x.GetIdx()
                for x in self.rdkit_mol.GetAtomsMatchingQuery(charge_equals_atoms)
            ]
        )
        charge_equals_atom_bool = np.in1d(self.atoms_array, charge_equals_atom)
        return charge_equals_atom_bool

    def charge_type_greater_than(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the FORMAL CHARGE is GREATER THAN the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        charge_greater_atoms = chemq.FormalChargeGreaterQueryAtom(number)
        charge_greater_atom = np.array(
            [
                x.GetIdx()
                for x in self.rdkit_mol.GetAtomsMatchingQuery(charge_greater_atoms)
            ]
        )
        charge_greater_atom_bool = np.in1d(self.atoms_array, charge_greater_atom)
        return charge_greater_atom_bool

    def hcount_less_than(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the HYDROGEN COUNT is LESS THAN the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        hcount_less_atoms = chemq.HCountLessQueryAtom(number)
        hcount_less_atom = np.array(
            [
                x.GetIdx()
                for x in self.rdkit_mol.GetAtomsMatchingQuery(hcount_less_atoms)
            ]
        )
        hcount_less_atom_bool = np.in1d(self.atoms_array, hcount_less_atom)
        return hcount_less_atom_bool

    def hcount_equals(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the HYDROGEN COUNT is EQUAL to the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        hcount_equals_atoms = chemq.HCountEqualsQueryAtom(number)
        hcount_equals_atom = np.array(
            [
                x.GetIdx()
                for x in self.rdkit_mol.GetAtomsMatchingQuery(hcount_equals_atoms)
            ]
        )
        hcount_equals_atom_bool = np.in1d(self.atoms_array, hcount_equals_atom)
        return hcount_equals_atom_bool

    def hcount_greater_than(self, number: int):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where the HYDROGEN COUNT is GREATER THAN the input.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        hcount_greater_atoms = chemq.HCountGreaterQueryAtom(number)
        hcount_greater_atom = np.array(
            [
                x.GetIdx()
                for x in self.rdkit_mol.GetAtomsMatchingQuery(hcount_greater_atoms)
            ]
        )
        hcount_greater_atom_bool = np.in1d(self.atoms_array, hcount_greater_atom)
        return hcount_greater_atom_bool

    def in_ring(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN A RING exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        ring_atoms = chemq.IsInRingQueryAtom()
        ring = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(ring_atoms)]
        )
        ring_bool = np.in1d(self.atoms_array, ring)
        return ring_bool

    def ring_size6(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN A 6-MEMBERED RING exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        ring_6 = chemq.MinRingSizeEqualsQueryAtom(6)
        size6 = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(ring_6)]
        )
        size6_bool = np.in1d(self.atoms_array, size6)
        return size6_bool

    def ring_size5(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN A 5-MEMBERED RING exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        ring_5 = chemq.MinRingSizeEqualsQueryAtom(5)
        size5 = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(ring_5)]
        )
        size5_bool = np.in1d(self.atoms_array, size5)
        return size5_bool

    def in_2_rings(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN 2 RINGS exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        ring_2 = chemq.InNRingsEqualsQueryAtom(2)
        ring2 = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(ring_2)]
        )
        ring2_bool = np.in1d(self.atoms_array, ring2)
        return ring2_bool

    def in_1_ring(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms IN 1 RING exist.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        ring_1 = chemq.InNRingsEqualsQueryAtom(1)
        ring1 = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(ring_1)]
        )
        ring1_bool = np.in1d(self.atoms_array, ring1)
        return ring1_bool

    def het_neighbors_3(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS = 3.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        het_a_3 = chemq.NumHeteroatomNeighborsEqualsQueryAtom(3)
        heta3 = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(het_a_3)]
        )
        heta3_bool = np.in1d(self.atoms_array, heta3)
        return heta3_bool

    def het_neighbors_2(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS = 2.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        het_a_2 = chemq.NumHeteroatomNeighborsEqualsQueryAtom(2)
        heta2 = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(het_a_2)]
        )
        heta2_bool = np.in1d(self.atoms_array, heta2)
        return heta2_bool

    def het_neighbors_1(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS = 1.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        het_a_1 = chemq.NumHeteroatomNeighborsEqualsQueryAtom(1)
        heta1 = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(het_a_1)]
        )
        heta1_bool = np.in1d(self.atoms_array, heta1)
        return heta1_bool

    def het_neighbors_0(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS = 0.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        het_a_0 = chemq.NumHeteroatomNeighborsEqualsQueryAtom(0)
        heta0 = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(het_a_0)]
        )
        heta0_bool = np.in1d(self.atoms_array, heta0)
        return heta0_bool

    def het_neighbors_greater_1(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS > 1.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        het_a_g1 = chemq.NumHeteroatomNeighborsGreaterQueryAtom(0)
        hetag1 = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(het_a_g1)]
        )
        hetag1_bool = np.in1d(self.atoms_array, hetag1)
        return hetag1_bool

    def het_neighbors_greater_0(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms have HETEROATOM NEIGHBORS > 0.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        het_a_g0 = chemq.NumHeteroatomNeighborsGreaterQueryAtom(0)
        hetag0 = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(het_a_g0)]
        )
        hetag0_bool = np.in1d(self.atoms_array, hetag0)
        return hetag0_bool

    def aliph_het_neighbors_2(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms are ALIPHATIC AND HAVE 2 HETEROATOM NEIGHBORS.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        a_het_a_2 = chemq.NumAliphaticHeteroatomNeighborsEqualsQueryAtom(2)
        aheta2 = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(a_het_a_2)]
        )
        aheta2_bool = np.in1d(self.atoms_array, aheta2)
        return aheta2_bool

    def aliph_het_neighbors_1(self):
        """
        This takes a numpy array of Atom IDs and returns a boolean for where atoms are ALIPHATIC AND HAS 1 HETEROATOM NEIGHBORS.
        Inputs to this function are built for an ORDERED LIST OF ALL ATOM IDs from LEAST TO GREATEST.
        """
        a_het_a_1 = chemq.NumAliphaticHeteroatomNeighborsEqualsQueryAtom(1)
        aheta1 = np.array(
            [x.GetIdx() for x in self.rdkit_mol.GetAtomsMatchingQuery(a_het_a_1)]
        )
        aheta1_bool = np.in1d(self.atoms_array, aheta1)
        return aheta1_bool

    def smarts_query(self, smarts: str):
        query = Chem.MolFromSmarts(smarts)
        substructs = self.rdkit_mol.GetSubstructMatches(query)

        idx = np.zeros(len(self.atoms), dtype=bool)
        for s in substructs:
            idx[list(s)] = True

        return idx
