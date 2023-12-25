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
Testing the descriptor calculation functionality
"""

import unittest as ut
import molli as ml


class DescriptorTC(ut.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_rectangular_grid(self):
        g1 = ml.descriptor.rectangular_grid([-1, 0, 0], [1, 0, 0], spacing=0.5)
        g2 = ml.descriptor.rectangular_grid([-1, -1, 0], [1, 1, 0], spacing=0.5)
        g3 = ml.descriptor.rectangular_grid([-1, -1, -1], [1, 1, 1], spacing=0.5)

        self.assertTupleEqual(g1.shape, (5, 3))
        self.assertTupleEqual(g2.shape, (25, 3))
        self.assertTupleEqual(g3.shape, (125, 3))

        g4 = ml.descriptor.rectangular_grid(
            [-1, -1, -1], [1, 1, 1], spacing=0.5, padding=0.1
        )
        self.assertTupleEqual(g4.shape, (125, 3))
