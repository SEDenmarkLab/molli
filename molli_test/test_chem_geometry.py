import unittest as ut

import molli as ml
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
        m = ml.CartesianGeometry.loads_xyz(H2O_XYZ)
        self.assertEqual(m.n_atoms, 3)
        self.assertTupleEqual(m.coords.shape, (3, 3))
        self.assertAlmostEqual(np.linalg.norm(m.coords - np.array(H2O_XYZ_LIST)), 0)
        self.assertAlmostEqual(m.distance(0, 1), 0.96900, places=5)
        self.assertAlmostEqual(m.distance(2, 1), 1.52694, places=5)
        self.assertEqual(m.formula, "H2 O1")

    def test_load_xyz(self):
        m1 = ml.CartesianGeometry.load_xyz(ml.files.xyz.pentane_confs.path)
        lm2 = ml.CartesianGeometry.load_all_xyz(ml.files.xyz.pentane_confs.path)
    
    @ut.skip("Not implemented yet")
    def test_loads_xyz(self):
        raise NotImplementedError
    
    @ut.skip("Not implemented yet")
    def test_dump_xyz(self):
        raise NotImplementedError

    @ut.skip("Not implemented yet")
    def test_dumps_xyz(self):
        raise NotImplementedError

    @ut.skip("Not implemented yet")
    def test_distance(self):
        raise NotImplementedError

    @ut.skip("Not implemented yet")
    def test_distance_to_point(self):
        raise NotImplementedError
    
    @ut.skip("Not implemented yet")
    def test_angle(self):
        raise NotImplementedError
    
    @ut.skip("Not implemented yet")
    def test_dihedral(self):
        raise NotImplementedError
