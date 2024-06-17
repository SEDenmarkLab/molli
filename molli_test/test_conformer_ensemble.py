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
Testing the conformer ensemble class functionality
"""

import unittest as ut
import numpy as np
import molli as ml

h2o_mol2_str = """@<TRIPOS>MOLECULE
    M0001
    3 2
    SMALL
    USER_CHARGES


    @<TRIPOS>ATOM
        1   H2         0.761229899    -0.478138566     0.000000000       H   1  M0001    0.376285
        2   O          0.000000000     0.120865773     0.000000000     O.3   1  M0001   -0.752569
        3   H1        -0.761229899    -0.478138566     0.000000000       H   1  M0001    0.376285


    @<TRIPOS>BOND
        1      1      2    1
        2      2      3    1"""


class ConformerEnsembleTC(ut.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_init_empty_constructor(self):
        "Tests calling ml.ConformerEnsemble() with no parameters"
        ens = ml.ConformerEnsemble()
        self.assertEqual(ens.n_conformers, 0)
        self.assertEqual(ens.n_atoms, 0)
        self.assertEqual(ens.name, "unknown")
        self.assertTupleEqual(ens.coords.shape, (0, 0, 3))
        self.assertTupleEqual(ens.weights.shape, (0,))
        self.assertTupleEqual(
            ens.atomic_charges.shape, (0, 0)
        )  # NOTE: changed dim from (0,) to (0,0)

    def test_init_simple_constructor(self):
        "Tests calling ml.ConformerEnsemble() with constructor arguments"
        # TODO: re-write?
        ens1 = ml.ConformerEnsemble(["O", "H", "H"])

        self.assertEqual(ens1.n_conformers, 0)
        self.assertEqual(ens1.n_atoms, 3)
        self.assertTupleEqual(ens1.coords.shape, (0, 3, 3))

        ens2 = ml.ConformerEnsemble(["O", "H", "H"], n_conformers=5)

        self.assertEqual(ens2.n_conformers, 5)
        self.assertEqual(ens2.n_atoms, 3)
        self.assertTupleEqual(ens2.coords.shape, (5, 3, 3))

    def test_init_with_other_empty_ensemble(self):
        "Tests initializing ConformerEnsemble from the other instance of ConformerEnsemble class"
        ens = ml.ConformerEnsemble(["O", "H", "H"])
        new_ens = ml.ConformerEnsemble(ens, name="new_ens")

        self.assertEqual(new_ens.n_conformers, ens.n_conformers)
        self.assertEqual(new_ens.n_atoms, ens.n_atoms)
        self.assertEqual(new_ens.name, "new_ens")
        self.assertTupleEqual(new_ens.coords.shape, ens.coords.shape)
        self.assertTupleEqual(new_ens.weights.shape, ens.weights.shape)
        self.assertTupleEqual(new_ens.atomic_charges.shape, ens.atomic_charges.shape)

    def test_init_with_other_ensemble(self):
        "Tests initializing ConformerEnsemble from the other instance of ConformerEnsemble class"
        ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
        new_ens = ml.ConformerEnsemble(ens, name="new_ens")

        self.assertEqual(new_ens.n_conformers, ens.n_conformers)
        self.assertEqual(new_ens.n_atoms, ens.n_atoms)
        self.assertEqual(new_ens.name, "new_ens")
        self.assertTupleEqual(new_ens.coords.shape, ens.coords.shape)
        self.assertTupleEqual(new_ens.weights.shape, ens.weights.shape)
        self.assertTupleEqual(new_ens.atomic_charges.shape, ens.atomic_charges.shape)

    def test_init_with_wrong_parameters(self):
        "Tests raising exceptions when trying to initialize ConformerEnsemble with erroneous parameters"
        with self.assertRaises(NotImplementedError):
            ens = ml.ConformerEnsemble([1, "H", 2.0])

        with self.assertRaises(ValueError):
            ens = ml.ConformerEnsemble([1, 120, -2])

        ens = ml.ConformerEnsemble(["O", "H", "H"])
        with self.assertRaises(ValueError):
            ens.atomic_charges = np.array([0.0, -1.0, -2.0, 4.0])

        with self.assertRaises(ValueError):
            ens = ml.ConformerEnsemble(["O", "H", "H"], n_conformers=-1)

    def test_from_single_molecule(self):
        "Tests initializing ConformerEnsemble from single molecule"
        ens = ml.ConformerEnsemble.loads_mol2(h2o_mol2_str)
        self.assertEqual(ens.name, "M0001")
        self.assertEqual(ens.n_conformers, 1)
        self.assertEqual(ens.n_atoms, 3)
        self.assertTupleEqual(ens.coords.shape, (1, 3, 3))
        self.assertTupleEqual(ens.weights.shape, (1,))
        self.assertTupleEqual(
            ens.atomic_charges.shape, (1, 3)
        )  # NOTE: changed dim from (3,) to (1,3)

    def test_init_with_molecule_list(self):
        "Tests initializing ConformerEnsemble from list of molecules"
        ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
        self.assertEqual(ens.n_conformers, 7)
        self.assertEqual(ens.n_atoms, 17)
        self.assertEqual(ens.n_bonds, 16)
        self.assertTupleEqual(ens.coords.shape, (7, 17, 3))
        self.assertTupleEqual(ens.weights.shape, (7,))
        self.assertTupleEqual(ens.atomic_charges.shape, (7, 17))

        charges = np.tile(
            np.array(
                [
                    -0.0653,
                    -0.0559,
                    0.023,
                    0.023,
                    0.023,
                    -0.0536,
                    0.0263,
                    0.0263,
                    -0.0559,
                    0.0265,
                    0.0265,
                    -0.0653,
                    0.0263,
                    0.0263,
                    0.023,
                    0.023,
                    0.023,
                ]
            ),
            (7, 1),
        )

        np.testing.assert_allclose(charges, ens.atomic_charges)

    def test_init_from_mol2_wrong_parameters(self):
        "Tests possible erroneous initialization of ConformerEnsemble"
        charges = np.array(
            [
                -0.0653,
                -0.0559,
                0.023,
                0.023,
                0.023,
                -0.0536,
                0.0263,
                0.0263,
                -0.0559,
                0.0265,
                0.0265,
                -0.0653,
                0.0263,
                0.0263,
                0.023,
                0.023,
                0.023,
            ]
        )

        with self.assertRaises(TypeError):
            ens = ml.ConformerEnsemble.load_mol2(
                ml.files.pentane_confs_mol2, atomic_charges=charges
            )

    def test_iterating_through_ensemble(self):
        "Test ability to iterate over instance of the ConformerEnsemble class"
        ens = ml.ConformerEnsemble(["O", "H", "H"], n_conformers=5)
        self.assertNotEqual(next(iter(ens)), None)

    def test_appending_to_ensemble(self):
        "Tests append() method of the the ConformerEnsemble class"
        ens = ml.ConformerEnsemble(["O", "H", "H"], n_conformers=1)
        prev_n_conf = ens.n_conformers
        struct = ml.Molecule(["O", "H", "H"])
        ens.append(struct)
        self.assertEqual(ens.n_conformers, prev_n_conf + 1)

    def test_extend_ensemble(self):
        "Tests extend() method of the the ConformerEnsemble class"
        ens1 = ml.ConformerEnsemble(["O", "H", "H"], n_conformers=2)
        ens1_n_conf = ens1.n_conformers
        ens2 = ml.ConformerEnsemble(["O", "H", "H"], n_conformers=5)
        ens1.extend(ens2)
        self.assertEqual(ens1.n_conformers, ens1_n_conf + ens2.n_conformers)

    # TODO: test_serialization and test_deserialization after Alex fixes the source code

    def test_scale(self):
        "Tests scaling coordinates of the conformer ensemble"
        ens = ml.ConformerEnsemble.loads_mol2(h2o_mol2_str)
        old_coords = ens.coords

        with self.assertRaises(ValueError):
            ens.scale(0.0)

        with self.assertRaises(ValueError):
            ens.scale(-0.5)

        ens.scale(0.5)
        ens.scale(2)
        self.assertTrue(np.allclose(ens.coords, old_coords))

    def test_invert(self):
        "Tests inverting the coordinates of the conformer ensemble"
        ens = ml.ConformerEnsemble.loads_mol2(h2o_mol2_str)
        old_coords = ens.coords
        ens.invert()
        ens.invert()
        self.assertTrue(np.allclose(ens.coords, old_coords))

    def test_ens_pickle_serialization(self):
        import pickle

        m1 = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
        m_pkl = pickle.dumps(m1)
        m2 = pickle.loads(m_pkl)

    def test_init_from_molecule(self):
        m = ml.Molecule.load_mol2(ml.files.dendrobine_mol2)
        m.charge = 1
        m.mult = 2

        ens = ml.ConformerEnsemble(m)
        self.assertEqual(ens.charge, 1)
        self.assertEqual(ens.mult, 2)
        self.assertEqual(ens.coords.shape, (1, 44, 3))
        self.assertEqual(ens.atomic_charges.shape, (1, 44))
        self.assertEqual(ens.weights.shape, (1,))

    def test_dumps(self):
        ens = ml.ConformerEnsemble.load_mol2(ml.files.pentane_confs_mol2)
        ens.dumps_xyz()
