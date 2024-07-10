# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by  Blake E. Ocampo
#               Casey L. Olen
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
Testing the external functionality related to RDKit
"""

import unittest as ut

import numpy as np
import molli as ml
import importlib.util
from molli.external import RDKitException, RDKitKekulizationException

def is_package_installed(pkg_name):
    return importlib.util.find_spec(pkg_name) is not None


class RDKitTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    @ut.skipUnless(is_package_installed("rdkit"), "RDKit is not installed")
    @ut.skipUnless(is_package_installed("IPython"), "IPython is not installed")
    @ut.skipUnless(is_package_installed("openbabel"), "Openbabel is not installed")
    def test_to_rdmol_zincdb(self):
        from rdkit.Chem.PropertyMol import PropertyMol
        from molli.external import rdkit as mrd

        with ml.files.zincdb_fda_mol2.open() as f:
            structs = ml.Structure.yield_from_mol2(f)
            for s in structs:
                ml_mol = ml.Molecule(s, name=s.name)
                #Not every Molecule is expected to work, it should either return an RDKit Exception or PropertyMol
                try:
                    rdmol = mrd.to_rdmol(ml_mol,via='sdf', remove_hs=True)
                    self.assertIsInstance(rdmol, PropertyMol)
                except Exception as e:
                    self.assertIsInstance(e, (RDKitException, RDKitKekulizationException))

    @ut.skipUnless(is_package_installed("rdkit"), "RDKit is not installed")
    @ut.skipUnless(is_package_installed("IPython"), "IPython is not installed")
    @ut.skipUnless(is_package_installed("openbabel"), "Openbabel is not installed")
    def test_to_rdmol(self):
        from rdkit.Chem.PropertyMol import PropertyMol
        from molli.external import rdkit as mrd

        mlmol = ml.Molecule.load_mol2(ml.files.dendrobine_mol2, name="dendrobine")

        mlmol.attrib['_MolTest'] = 1
        for a in mlmol.atoms:
            a.attrib['_AtomTest'] = 1

        #Mol2 Test Keep Hydrogens
        rdmol = mrd.to_rdmol(mlmol,via='mol2', remove_hs=False, set_atts=True)
        self.assertIsInstance(rdmol, PropertyMol)
        self.assertTrue(rdmol.HasProp('_MolTest'))
        for a in rdmol.GetAtoms():
            self.assertTrue(a.HasProp('_AtomTest'))
        #Mol2 Test Remove Hydrogens
        rdmol = mrd.to_rdmol(mlmol,via='mol2', remove_hs=True, set_atts=True)
        self.assertIsInstance(rdmol, PropertyMol)
        self.assertTrue(rdmol.HasProp('_MolTest'))
        for a in rdmol.GetAtoms():
            self.assertFalse(a.HasProp('_AtomTest'))

        #XYZ Test Keep Hydrogens
        rdmol = mrd.to_rdmol(mlmol,via='xyz', remove_hs=False, set_atts=True)
        self.assertIsInstance(rdmol, PropertyMol)
        self.assertTrue(rdmol.HasProp('_MolTest'))
        for a in rdmol.GetAtoms():
            self.assertTrue(a.HasProp('_AtomTest'))
        #XYZ Test Remove Hydrogens
        rdmol = mrd.to_rdmol(mlmol,via='xyz', remove_hs=True, set_atts=True)
        self.assertIsInstance(rdmol, PropertyMol)
        self.assertTrue(rdmol.HasProp('_MolTest'))
        for a in rdmol.GetAtoms():
            self.assertFalse(a.HasProp('_AtomTest'))

        #MOL/SDF Test Keep Hydrogens
        rdmol = mrd.to_rdmol(mlmol,via='mol', remove_hs=False, set_atts=True)
        self.assertIsInstance(rdmol, PropertyMol)
        self.assertTrue(rdmol.HasProp('_MolTest'))
        for a in rdmol.GetAtoms():
            self.assertTrue(a.HasProp('_AtomTest'))
        #MOL/SDF Test Remove Hydrogens
        rdmol = mrd.to_rdmol(mlmol,via='mol', remove_hs=True, set_atts=True)
        self.assertIsInstance(rdmol, PropertyMol)
        self.assertTrue(rdmol.HasProp('_MolTest'))
        for a in rdmol.GetAtoms():
            self.assertFalse(a.HasProp('_AtomTest'))

        #PDB Test Keep Hydrogens
        rdmol = mrd.to_rdmol(mlmol,via='pdb', remove_hs=False, set_atts=True)
        self.assertIsInstance(rdmol, PropertyMol)
        self.assertTrue(rdmol.HasProp('_MolTest'))
        for a in rdmol.GetAtoms():
            self.assertTrue(a.HasProp('_AtomTest'))
        #PDB Test Remove Hydrogens
        rdmol = mrd.to_rdmol(mlmol,via='pdb', remove_hs=True, set_atts=True)
        self.assertIsInstance(rdmol, PropertyMol)

        #SMILES Test Keep Hydrogens
        rdmol = mrd.to_rdmol(mlmol,via='smi', remove_hs=False, set_atts=True)
        self.assertIsInstance(rdmol, PropertyMol)
        self.assertTrue(rdmol.HasProp('_MolTest'))
        #This molecule maintains the order, so it should be True
        for a in rdmol.GetAtoms():
            self.assertTrue(a.HasProp('_AtomTest'))
        #SMILES Test Remove Hydrogens
        rdmol = mrd.to_rdmol(mlmol,via='smi', remove_hs=True, set_atts=True)
        self.assertIsInstance(rdmol, PropertyMol)
        self.assertTrue(rdmol.HasProp('_MolTest'))
        for a in rdmol.GetAtoms():
            self.assertFalse(a.HasProp('_AtomTest'))

    @ut.skipUnless(is_package_installed("rdkit"), "RDKit is not installed")
    @ut.skipUnless(is_package_installed("IPython"), "IPython is not installed")
    @ut.skipUnless(is_package_installed("openbabel"), "Openbabel is not installed")
    def test_ml_mol_reorder(self):
        from rdkit.Chem.PropertyMol import PropertyMol
        from molli.external import rdkit as mrd

        with ml.files.zincdb_fda_mol2.open() as f:
            structs = ml.Structure.yield_from_mol2(f)
            for s in structs:
                ml_mol = ml.Molecule(s, name=s.name)
                #Not every molecule is readable within the ZincDB, this is just testing molecules
                try:
                    rdmol = mrd.to_rdmol(ml_mol,via='sdf', remove_hs=False)
                except:
                    continue
                else:
                    if isinstance(rdmol, PropertyMol):
                        rd_can_mol, atom_reorder, bond_reorder = mrd.can_mol_order(
                            rdmol
                        )
                        molli_can_rdkit_dict = mrd.reorder_molecule(
                            ml_mol,
                            can_rdmol_w_h=rd_can_mol,
                            can_atom_reorder=atom_reorder,
                            can_bond_reorder=bond_reorder,
                        )
                        for ml_mol, rdkit_mol in molli_can_rdkit_dict.items():
                            can_rdkit_atom_elem = np.array(
                                [x.GetSymbol() for x in rdkit_mol.GetAtoms()]
                            )
                            new_molli_elem = np.array(
                                [atom.element.symbol for atom in ml_mol.atoms]
                            )
                            np.testing.assert_array_equal(can_rdkit_atom_elem, new_molli_elem)
                


    @ut.skipUnless(is_package_installed("rdkit"), "RDKit is not installed")
    @ut.skipUnless(is_package_installed("IPython"), "IPython is not installed")
    @ut.skipUnless(is_package_installed("openbabel"), "Openbabel is not installed")
    def test_atom_filter(self):
        from molli.external import rdkit as mrd

        m1 = ml.Molecule.load_mol2(ml.files.dendrobine_mol2, name="dendrobine")

        rdmol = mrd.to_rdmol(m1,via='sdf', remove_hs=False)
        maf_mol = mrd.atom_filter(rdmol)

        af_mol_sp2_bool = maf_mol.sp2_type()
        num_sp2_atoms = np.count_nonzero(af_mol_sp2_bool)
        self.assertEqual(num_sp2_atoms, 3)

        af_mol_het_neighbors_2 = maf_mol.het_neighbors_2()
        num_het_neighbors_2 = np.count_nonzero(af_mol_het_neighbors_2)
        self.assertEqual(num_het_neighbors_2, 1)

        af_mol_dual_bool = maf_mol.sp2_type() & maf_mol.het_neighbors_2()
        num_af_mol_dual_bool = np.count_nonzero(af_mol_dual_bool)
        self.assertEqual(num_af_mol_dual_bool, 1)
