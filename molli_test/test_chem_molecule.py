# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by  Alexander S. Shved
#               Casey L. Olen
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
Testing the `Molecule` class functionality
"""

import unittest as ut
import numpy as np
from pathlib import Path

import molli as ml


class MoleculeTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_molecule_empty_constructor(self):
        m1 = ml.Molecule()
        m2 = ml.Molecule()

        self.assertIsInstance(m1, ml.Molecule)  # make sure its a molecule

        self.assertEqual(len(m1.atoms), 0)  # no atoms
        self.assertEqual(len(m1.atomic_charges), 0)  # no charges
        self.assertEqual(len(m1.bonds), 0)  # no bonds
        self.assertEqual(m1.coords.shape, (0, 3))  # no coords, but correct shape
        self.assertEqual(m1.charge, 0)  # total charge is 0
        self.assertEqual(m1.mult, 1)  # multiplicity is 1, closed shell
        self.assertEqual(m1.name, "unknown")  # name is unknown

        self.assertNotEqual(id(m1), id(m2))  # should be different objects in memory
        self.assertNotEqual(id(m1.atoms), id(m2.atoms))  # atom list should be different
        self.assertNotEqual(
            id(m1.atomic_charges), id(m2.atomic_charges)
        )  # atom list should be different
        self.assertNotEqual(id(m1.bonds), id(m2.bonds))  # bonds should be different
        self.assertNotEqual(id(m1.coords), id(m2.coords))  # coords should be different

        self.assertEqual(
            id(m1.charge), id(m2.charge)
        )  # primitive types should be the same
        self.assertEqual(id(m1.mult), id(m2.mult))  # primitive types should be the same
        self.assertEqual(id(m1.name), id(m2.name))

    def test_yield_from_mol2(self):
        with open(ml.files.pentane_confs_mol2) as f:
            lst_structs: list[ml.Molecule] = list(
                ml.Molecule.yield_from_mol2(f, name="pentane")
            )

        self.assertEqual(len(lst_structs), 7)

        for m in lst_structs:
            self.assertEqual(m.n_atoms, 17)
            self.assertEqual(m.n_bonds, 16)

    # @ut.skip("Not implemented yet")
    def test_molecule_cloning(self):
        with open(ml.files.pentane_confs_mol2) as f:
            m1 = ml.Molecule.load_mol2(f, name="pentane")

        m2 = ml.Molecule(m1)

        self.assertNotEqual(id(m1._atoms), id(m2._atoms))
        self.assertNotEqual(id(m1._bonds), id(m2._bonds))

        for a1, a2 in zip(m1.atoms, m2.atoms):
            self.assertNotEqual(a1, a2)

        for b1, b2 in zip(m1.bonds, m2.bonds):
            self.assertNotEqual(b1, b2)

        self.assertNotEqual(id(m1._coords), id(m2._coords))

        self.assertEqual(np.linalg.norm(m1.coords - m2.coords), 0)

    def test_load_mol2_charges(self):
        mol = ml.load(ml.files.isornitrate_mol2, parser="molli")

        self.assertEqual(mol.atoms[13].formal_charge, 1)
        self.assertEqual(mol.atoms[17].formal_charge, -1)
        self.assertEqual(mol.charge, 0)

    def test_dump_mol2(self):
        for file in [
            ml.files.pentane_confs_mol2,
            ml.files.dendrobine_mol2,
            ml.files.dummy_mol2,
            ml.files.fxyl_mol2,
            ml.files.nanotube_mol2,
        ]:
            with open(file) as f:
                m1 = ml.Molecule.load_mol2(f)

            new_m1 = next(ml.Molecule.yield_from_mol2(ml.Molecule.dumps_mol2(m1)))

            np.testing.assert_array_equal(m1._coords, new_m1._coords)

            self.assertListEqual(
                [a1.element.symbol for a1 in m1.atoms],
                [new_a1.element.symbol for new_a1 in new_m1.atoms],
            )

            self.assertListEqual(
                [b1.btype for b1 in m1.bonds], [new_b1.btype for new_b1 in new_m1.bonds]
            )

    def test_del_atom(self):
        """This tests if atom deletion is correctly handled for atomic charges"""

        mol = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
        na = mol.n_atoms
        nb = mol.n_bonds

        mol.del_atom(0)

        self.assertEqual(mol.n_atoms, na - 1)
        self.assertEqual(mol.n_bonds, nb - 3)  # atom 0 is connected to 3 bonds
        self.assertEqual(mol._atomic_charges.shape, (na - 1,))

    def test_add_atom(self):
        """This tests if atom addition is correctly handled for atomic charges, coordinates"""

        mol = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
        na = mol.n_atoms
        nb = mol.n_bonds
        ncoords = mol.coords.shape

        a = ml.Atom()
        mol.add_atom(a, [0, 0, 0])

        self.assertEqual(mol.n_atoms, na + 1)
        self.assertEqual(mol.n_bonds, nb)  # added an atom but no bonds
        self.assertEqual(mol._atomic_charges.shape, (na + 1,))
        self.assertEqual(mol.coords.shape, (ncoords[0] + 1, 3))

    def test_parent_property(self):
        mol = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)

        for a in mol.atoms:
            assert a.parent is mol

        for b in mol.bonds:
            assert b.parent is mol

    def test_mol_pickle_serialization(self):
        import pickle

        m1 = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
        m_pkl = pickle.dumps(m1)
        m2 = pickle.loads(m_pkl)
