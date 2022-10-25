import unittest as ut

from molli import chem
import numpy as np


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


class GeometryTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_xyzimport(self):
        m = chem.CartesianGeometry.from_xyz(H2O_XYZ)
        self.assertEqual(m.n_atoms, 3)
        self.assertTupleEqual(m.coords.shape, (3, 3))
        self.assertAlmostEqual(np.linalg.norm(m.coords - np.array(H2O_XYZ_LIST)), 0)
        self.assertAlmostEqual(m.distance(0, 1), 0.96900, places=5)
        self.assertAlmostEqual(m.distance(2, 1), 1.52694, places=5)
        self.assertEqual(m.formula, "H2 O1")

    ##### This is likely not needed anymore #####

    # def test_coords_protected(self):
    #     m = chem.CartesianGeometry.from_xyz(H2O_XYZ)
    #     with self.assertRaises(AttributeError):
    #         m.coords = [1, 2, 3]
