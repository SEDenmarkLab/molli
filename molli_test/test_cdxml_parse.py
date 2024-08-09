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

# formula, charge, multiplicity
CHARGE_MULT_ANSWER = {
    "a1": ("C11 H18 N1", 1, 1),
    "a2": ("C9 H18 N1 O1", 0, 2),
    "a3": ("C6 H11 N1 O1", 0, 1),
    "a4": ("C6 H11 N1 O1", 0, 1),
    "a5": ("C6 H5 N1 O2", 0, 1),
    "a6": ("C8 H19 N1", 1, 2),
    "a7": ("C6 H8 O1", 0, 3),
    "a8": ("C1 Cl2", 0, 3),
    "a9": ("C2 H4 B1 O2", 1, 1),
    "a10": ("C4 H10 P1", 1, 1),
    "a11": ("C1 H6 B1", -1, 1),
}


class CDXMLParserTC(ut.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.charges_mult_cdxml = ml.CDXMLFile(ml.files.charges_mult_cdxml)

    def test_charge_mult_parsing(self):
        for mol_id in CHARGE_MULT_ANSWER:
            mol = self.charges_mult_cdxml[mol_id]
            mol.add_implicit_hydrogens()

            formula_correct, charge_correct, mult_correct = CHARGE_MULT_ANSWER[mol_id]
            formula_parsed, charge_parsed, mult_parsed = (
                mol.formula,
                mol.charge,
                mol.mult,
            )
            self.assertEqual(
                formula_correct,
                formula_parsed,
                f"Formula correct != parsed for molecule {mol.name!r}",
            )
            self.assertEqual(
                charge_correct,
                charge_parsed,
                f"Charge correct != parsed for molecule {mol.name!r}",
            )
            self.assertEqual(
                mult_correct,
                mult_parsed,
                f"Multiplicity correct != parsed for molecule {mol.name!r}",
            )
