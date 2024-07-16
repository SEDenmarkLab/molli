# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Alexander S. Shved <shvedalx@illinois.edu>
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
Testing the `Structure` class functionality
"""

import unittest as ut

import molli as ml
import numpy as np
from pathlib import Path


H2O_XYZ = """3
water molecule, in angstrom
O        0.0000000000      0.0000000000      0.0000000000                 
H       -0.7634678742     -0.2331373366      0.5492948095                 
H        0.7634678742     -0.2331373366      0.5492948095
"""

H2O_XYZ_LIST = [
    [0.0000000000, 0.0000000000, 0.0000000000],
    [-0.7634678742, -0.2331373366, 0.5492948095],
    [0.7634678742, -0.2331373366, 0.5492948095],
]


class StructureTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_structure_3d(self):
        m = ml.Structure.loads_xyz(H2O_XYZ, name="water", source_units="Angstrom")
        self.assertEqual(m.n_atoms, 3)
        self.assertTupleEqual(m.coords.shape, (3, 3))
        self.assertAlmostEqual(np.linalg.norm(m.coords - np.array(H2O_XYZ_LIST)), 0)
        self.assertAlmostEqual(m.distance(0, 1), 0.96900, places=5)
        self.assertAlmostEqual(m.distance(2, 1), 1.52694, places=5)
        self.assertEqual(m.formula, "H2 O1")
        self.assertEqual(m.name, "water")

    def test_structure_from_xyz_file(self):
        # This opens in text mode
        with open(ml.files.dendrobine_xyz) as f:
            m1 = ml.Structure.load_xyz(f, name="dendrobine", source_units="Angstrom")

        m2 = ml.Structure(m1)
        # This just makes sure that
        self.assertTrue(np.allclose(m1.coords, m2.coords, atol=1e-5))

    def test_yield_structures_from_xyz_file(self):
        with open(ml.files.pentane_confs_xyz) as f:
            lst_structs = list(ml.Structure.yield_from_xyz(f, name="pentane"))

        self.assertEqual(len(lst_structs), 7)

    def test_add_bonds(self):
        m = ml.Structure.loads_xyz(H2O_XYZ)
        self.assertEqual(m.n_bonds, 0)
        b1 = ml.Bond(m.atoms[0], m.atoms[2])
        b2 = ml.Bond(m.atoms[0], m.atoms[1])
        m.append_bonds(b1, b2)
        self.assertEqual(m.n_bonds, 2)

    def test_concatenate(self):
        s1 = ml.Structure.load_mol2(ml.files.dendrobine_mol2)

        s2 = ml.Structure(s1)
        s2.translate([50, 0, 0])

        s3 = s1 | s2

        self.assertEqual(s3.n_atoms, s1.n_atoms * 2)
        self.assertEqual(s3.n_bonds, s1.n_bonds * 2)

        self.assertAlmostEqual(s3.distance(0, s1.n_atoms), 50.0, 6)

    def test_zincdb(self):
        with open(ml.files.dendrobine_mol2) as f:
            structs = ml.Structure.yield_from_mol2(f)

            for s in structs:
                self.assertIsNotNone(s.name)
                self.assertIsNot(s.name, "unnamed")
                self.assertTrue(all(a.label is not None for a in s.atoms))

    def test_load_mol2_dummy(self):
        m = ml.Structure.load_mol2(ml.files.dummy_mol2)
        a1, a2 = m.atoms
        self.assertEqual(a1.atype, ml.AtomType.Dummy)
        self.assertEqual(a2.atype, ml.AtomType.Dummy)

        names = [n for n in m.name]

        self.assertFalse(any(n == "unnamed" for n in names))

    def test_substruct(self):
        s1 = ml.Structure.load_mol2(ml.files.dendrobine_mol2)
        a_test = (1, 3, 5)
        sub = ml.Substructure(s1, a_test)
        sub_coord = sub.coords
        s1_coord = s1.coord_subset(a_test)

        np.testing.assert_allclose(s1_coord, sub_coord)

        pro1 = ml.Promolecule(s1)
        pro1.atoms

        con1 = ml.Connectivity(s1)
        con1.atoms
        con1.bonds

        struct1 = ml.Structure(s1)
        struct1.atoms
        struct1.bonds

        m1 = ml.Molecule(s1)
        m1.atoms
        m1.bonds

    def test_del_atom(self):
        s1 = ml.Structure.load_mol2(ml.files.dendrobine_mol2)
        s1.del_atom(0)

        self.assertEqual(s1.n_atoms, 43)
        self.assertEqual(s1.n_bonds, 44)
