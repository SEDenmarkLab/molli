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
Testing the `CartesianGeometry` class functionality
"""

import unittest as ut
import numpy as np
import molli as ml
import math


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


try:
    WATER = ml.CartesianGeometry.loads_xyz(H2O_XYZ)
except:
    WATER = None

try:
    PENTANE = ml.CartesianGeometry.loads_xyz(H2O_XYZ)
except:
    PENTANE = None


class GeometryTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_loads_xyz(self):
        m = ml.CartesianGeometry.loads_xyz(H2O_XYZ)
        self.assertEqual(m.n_atoms, 3)
        self.assertTupleEqual(m.coords.shape, (3, 3))
        self.assertAlmostEqual(np.linalg.norm(m.coords - np.array(H2O_XYZ_LIST)), 0)
        self.assertAlmostEqual(m.distance(0, 1), 0.96900, places=5)
        self.assertAlmostEqual(m.distance(2, 1), 1.52694, places=5)
        self.assertEqual(m.formula, "H2 O1")

    def test_load_xyz(self):
        m1 = ml.CartesianGeometry.load_xyz(ml.files.pentane_confs_xyz)
        lm2 = ml.CartesianGeometry.load_all_xyz(ml.files.pentane_confs_xyz)

    def test_load_xyz_dummy(self):
        m = ml.CartesianGeometry.load_xyz(ml.files.dummy_xyz)
        a1, a2 = m.atoms
        self.assertEqual(a1.atype, ml.AtomType.Dummy)
        self.assertEqual(a2.atype, ml.AtomType.Dummy)

    @ut.skip("Not implemented yet")
    def test_dump_xyz(self):
        raise NotImplementedError

    @ut.skip("Not implemented yet")
    def test_dumps_xyz(self):
        raise NotImplementedError

    def test_distance(self):
        d1 = WATER.distance(0, 1)
        d2 = WATER.distance(0, 2)

        self.assertAlmostEqual(d1, d2, 6)
        self.assertAlmostEqual(d1, math.dist(H2O_XYZ_LIST[0], H2O_XYZ_LIST[1]), 6)

    def test_distance_to_point(self):
        d = WATER.distance_to_point(0, [50, 0, 0])
        self.assertAlmostEqual(d, 50, 6)

    @ut.skip("Not implemented yet")
    def test_angle(self):
        raise NotImplementedError

    @ut.skip("Not implemented yet")
    def test_dihedral(self):
        raise NotImplementedError
