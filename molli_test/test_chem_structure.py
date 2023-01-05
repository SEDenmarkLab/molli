import unittest as ut

import molli as ml
from molli import chem
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
        m = chem.Structure.from_xyz(H2O_XYZ, name="water", source_units="Angstrom")
        self.assertEqual(m.n_atoms, 3)
        self.assertTupleEqual(m.coords.shape, (3, 3))
        self.assertAlmostEqual(np.linalg.norm(m.coords - np.array(H2O_XYZ_LIST)), 0)
        self.assertAlmostEqual(m.distance(0, 1), 0.96900, places=5)
        self.assertAlmostEqual(m.distance(2, 1), 1.52694, places=5)
        self.assertEqual(m.formula, "H2 O1")
        self.assertEqual(m.name, "water")

    def test_structure_from_xyz_file(self):
        # This opens in text mode
        with ml.files.xyz.dendrobine.open() as f:
            m1 = chem.Structure.from_xyz(f, name="dendrobine", source_units="Angstrom")

        m2 = chem.Structure(m1)
        # This just makes sure that
        self.assertTrue(np.allclose(m1.coords, m2.coords, atol=1e-5))

    def test_yield_structures_from_xyz_file(self):
        with ml.files.xyz.pentane_confs.open() as f:
            lst_structs = list(chem.Structure.yield_from_xyz(f, name="pentane"))

        self.assertEqual(len(lst_structs), 7)

    def test_add_bonds(self):
        m = chem.Structure.from_xyz(H2O_XYZ)
        self.assertEqual(m.n_bonds, 0)
        b1 = chem.Bond(m.atoms[0], m.atoms[2])
        b2 = chem.Bond(m.atoms[0], m.atoms[1])
        m.append_bonds(b1, b2)
        self.assertEqual(m.n_bonds, 2)

    def test_concatenate(self):
        with ml.files.mol2.dendrobine.open() as f:
            s1 = chem.Structure.from_mol2(f)

        s2 = chem.Structure(s1)
        s2.translate([50, 0, 0])

        s3 = s1 | s2

        self.assertEqual(s3.n_atoms, s1.n_atoms * 2)
        self.assertEqual(s3.n_bonds, s1.n_bonds * 2)

        self.assertAlmostEqual(s3.distance(0, s1.n_atoms), 50.0, 6)
