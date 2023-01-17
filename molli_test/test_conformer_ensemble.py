import unittest as ut
import numpy as np
import molli as ml


class ConformerEnsembleTC(ut.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_simple_constructor(self):
        ens1 = ml.ConformerEnsemble(["O", "H", "H"])

        self.assertEqual(ens1.n_conformers, 0)
        self.assertEqual(ens1.n_atoms, 3)
        self.assertTupleEqual(ens1.coords.shape, (0, 3, 3))

        ens2 = ml.ConformerEnsemble(["O", "H", "H"], n_conformers=5)

        self.assertEqual(ens2.n_conformers, 5)
        self.assertEqual(ens2.n_atoms, 3)
        self.assertTupleEqual(ens2.coords.shape, (5, 3, 3))

    def test_pentane_conformers(self):
        ens = ml.ConformerEnsemble.load_mol2(ml.files.mol2.pentane_confs.path)
        self.assertEqual(ens.n_conformers, 7)
        self.assertEqual(ens.n_atoms, 17)
        self.assertEqual(ens.n_bonds, 16)
        self.assertTupleEqual(ens.coords.shape, (7, 17, 3))
        self.assertTupleEqual(ens.weights.shape, (7,))
        self.assertTupleEqual(ens.atomic_charges.shape, (17,))

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

        np.testing.assert_allclose(
            charges, ens.atomic_charges
        )
