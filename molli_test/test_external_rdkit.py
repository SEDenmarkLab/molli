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


def is_package_installed(pkg_name):
    return importlib.util.find_spec(pkg_name) is not None


class RDKitTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    @ut.skipUnless(is_package_installed("rdkit"), "RDKit is not installed")
    @ut.skipUnless(is_package_installed("IPython"), "IPython is not installed")
    @ut.skipUnless(is_package_installed("openbabel"), "Openbabel is not installed")
    def test_create_rdkit_mol(self):
        from rdkit.Chem.PropertyMol import PropertyMol
        from molli.external import _rdkit as mrd

        with ml.files.zincdb_fda_mol2.open() as f:
            structs = ml.Structure.yield_from_mol2(f)
            for s in structs:
                molli_mol = ml.Molecule(s, name=s.name)
                molli_mol_rdmol_dict = mrd.create_rdkit_mol(molli_mol)
                self.assertIsInstance(molli_mol_rdmol_dict[molli_mol], PropertyMol)

    @ut.skipUnless(is_package_installed("rdkit"), "RDKit is not installed")
    @ut.skipUnless(is_package_installed("IPython"), "IPython is not installed")
    @ut.skipUnless(is_package_installed("openbabel"), "Openbabel is not installed")
    def test_molli_mol_reorder(self):
        from rdkit.Chem.PropertyMol import PropertyMol
        from molli.external import _rdkit as mrd

        with ml.files.zincdb_fda_mol2.open() as f:
            structs = ml.Structure.yield_from_mol2(f)
            for s in structs:
                molli_mol = ml.Molecule(s, name=s.name)
                molli_mol_rdmol_dict = mrd.create_rdkit_mol(molli_mol)
                rd_can_mol, atom_reorder, bond_reorder = mrd.can_mol_order(
                    molli_mol_rdmol_dict[molli_mol]
                )
                molli_can_rdkit_dict = mrd.reorder_molecule(
                    molli_mol,
                    can_rdkit_mol_w_h=rd_can_mol,
                    can_atom_reorder=atom_reorder,
                    can_bond_reorder=bond_reorder,
                )
                for molli_mol, rdkit_mol in molli_can_rdkit_dict.items():
                    can_rdkit_atom_elem = np.array(
                        [x.GetSymbol() for x in rdkit_mol.GetAtoms()]
                    )
                    new_molli_elem = np.array(
                        [atom.element.symbol for atom in molli_mol.atoms]
                    )
                    np.testing.assert_array_equal(can_rdkit_atom_elem, new_molli_elem)

    @ut.skipUnless(is_package_installed("rdkit"), "RDKit is not installed")
    @ut.skipUnless(is_package_installed("IPython"), "IPython is not installed")
    @ut.skipUnless(is_package_installed("openbabel"), "Openbabel is not installed")
    def test_rdkit_atom_filter(self):
        from rdkit.Chem.PropertyMol import PropertyMol
        from molli.external import _rdkit as mrd

        m1 = ml.Molecule.load_mol2(ml.files.dendrobine_mol2, name="dendrobine")

        molli_rdkit_dict = mrd.create_rdkit_mol(m1)
        af_mol = mrd.rdkit_atom_filter(molli_rdkit_dict[m1])

        af_mol_sp2_bool = af_mol.sp2_type()
        num_sp2_atoms = np.count_nonzero(af_mol_sp2_bool)
        self.assertEqual(num_sp2_atoms, 3)

        af_mol_het_neighbors_2 = af_mol.het_neighbors_2()
        num_het_neighbors_2 = np.count_nonzero(af_mol_het_neighbors_2)
        self.assertEqual(num_het_neighbors_2, 1)

        af_mol_dual_bool = af_mol.sp2_type() & af_mol.het_neighbors_2()
        num_af_mol_dual_bool = np.count_nonzero(af_mol_dual_bool)
        self.assertEqual(num_af_mol_dual_bool, 1)
