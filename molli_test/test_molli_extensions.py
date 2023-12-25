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
Testing the functionality of molli extensions
"""

import unittest as ut
import numpy as np
import scipy.spatial.distance as spd
import molli_xt


class ExtensionTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_cdist3f(self):
        a1 = np.random.rand(5000, 3).astype(np.float32)
        a2 = np.random.rand(100, 3).astype(np.float32)

        d_molli = molli_xt.cdist22_eu2(a1, a2)
        d_spd = spd.cdist(a1, a2, metric="sqeuclidean")

        self.assertAlmostEqual(np.max(d_molli - d_spd), 0, places=6)
