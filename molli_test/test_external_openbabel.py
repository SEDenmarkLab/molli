# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Blake E. Ocampo
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
Testing the external functionality related to `OpenBabel`
"""

import unittest as ut

import molli as ml
import numpy as np
from copy import deepcopy
import importlib.util


def is_package_installed(pkg_name):
    return importlib.util.find_spec(pkg_name) is not None


class OpenbabelTC(ut.TestCase):
    @ut.skipUnless(is_package_installed("openbabel"), "Openbabel is not installed")
    def test_ob_mol2(self):
        from molli.external import openbabel as mob

        with ml.files.dendrobine_mol2.open() as f:
            mlmol = ml.Molecule.load_mol2(f)

        mol2_block = mob.to_mol2_w_ob(mlmol)

        new_mlmol = next(ml.Structure.yield_from_mol2(mol2_block))

        self.assertEqual(len(mlmol.atoms), len(new_mlmol.atoms))
        self.assertEqual(len(mlmol.bonds), len(new_mlmol.bonds))

    @ut.skipUnless(is_package_installed("openbabel"), "Openbabel is not installed")
    def test_ob_opt(self):
        from molli.external import openbabel as mob

        with ml.files.dendrobine_mol2.open() as f:
            mlmol = next(ml.Molecule.yield_from_mol2(f))

        old_coords = deepcopy(mlmol.coords)

        opt_mlmol = mob.obabel_optimize(mlmol, ff="UFF")

        opt_coords = opt_mlmol.coords
        self.assertTupleEqual(old_coords.shape, opt_coords.shape)

        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, old_coords, opt_coords
        )
