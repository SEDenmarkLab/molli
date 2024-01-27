# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by  Blake E. Ocampo
#               Casey L. Olen
#               Alexander S. Shved
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
Testing the external functionality related to XTB
"""

import unittest as ut

import numpy as np
import molli as ml
import subprocess
import os
from pathlib import Path
import shutil

from joblib import delayed, Parallel
from molli.pipeline.xtb import XTBDriver
from molli.config import BACKUP_DIR
from molli.config import SCRATCH_DIR


# a little clunky but will check if xTB is installed
try:
    out = subprocess.run(["xtb", "--version"], capture_output=True)
    if out.stderr == b"normal termination of xtb\n":
        _XTB_INSTALLED = True
except FileNotFoundError:
    _XTB_INSTALLED = False

_CURRENT_BACKUP_DIR = BACKUP_DIR
_CURRENT_SCRATCH_DIR = SCRATCH_DIR
_TEST_BACKUP_DIR: Path = ml.config.HOME / "test_backup"
_TEST_SCRATCH_DIR: Path = ml.config.HOME / "test_scratch"


def prep_dirs():
    ml.config.BACKUP_DIR = _TEST_BACKUP_DIR
    ml.config.SCRATCH_DIR = _TEST_SCRATCH_DIR
    # _TEST_BACKUP_DIR.mkdir()
    # _TEST_SCRATCH_DIR.mkdir()


def cleanup_dirs():
    shutil.rmtree(_TEST_BACKUP_DIR)
    shutil.rmtree(_TEST_SCRATCH_DIR)
    ml.config.BACKUP_DIR = _CURRENT_BACKUP_DIR
    ml.config.SCRATCH_DIR = _CURRENT_SCRATCH_DIR


class XTBTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    @ut.skipUnless(_XTB_INSTALLED, "xtb is not installed in current environment.")
    def test_xtb_optimize_series(self):
        source = ml.MoleculeLibrary(ml.files.cinchonidine_no_conf)
        # test with cinchonidine library
        # with ml.MoleculeLibrary.reading(ml.files.cinchonidines) as source:
        with source.reading():
            prep_dirs()
            xtb = XTBDriver(nprocs=4)
            res = [xtb.optimize(source[m_name]) for m_name in source]

            for m1, m2 in zip([source[m_name] for m_name in source], res):
                self.assertEqual(m1.name, m2.name, "Names must be the same!")
                self.assertNotAlmostEqual(
                    np.linalg.norm(m1.coords - m2.coords), 0
                )  # make sure the atom coordinates have moved
            cleanup_dirs()

    @ut.skipUnless(_XTB_INSTALLED, "xtb is not installed in current environment.")
    def test_xtb_optimize_parallel(self):
        source = ml.MoleculeLibrary(ml.files.cinchonidine_no_conf)
        # test with cinchonidine library
        with source.reading():
            prep_dirs()

            xtb = XTBDriver(nprocs=4)
            res = Parallel(n_jobs=32, verbose=50, prefer="threads")(
                delayed(xtb.optimize)(
                    M=source[m_name],
                    method="gff",
                )
                for m_name in source
            )
            for m1, m2 in zip([source[m_name] for m_name in source], res):
                self.assertEqual(m1.name, m2.name, "Names must be the same!")
                self.assertNotAlmostEqual(
                    np.linalg.norm(m1.coords - m2.coords), 0
                )  # make sure the atom coordinates have moved
            cleanup_dirs()

    @ut.skipUnless(_XTB_INSTALLED, "xtb is not installed in current environment.")
    def test_xtb_energy_series(self):
        source = ml.MoleculeLibrary(ml.files.cinchonidine_no_conf)
        # test with cinchonidine library
        # with ml.MoleculeLibrary.reading(ml.files.cinchonidines) as source:
        with source.reading():
            prep_dirs()
            xtb = XTBDriver(nprocs=4)
            res = [xtb.energy(source[m_name]) for m_name in source]

            # we will spot check several of these based on output from separate call to xtb
            for i, name in enumerate(source):
                if name == "1_5_c_cf0":
                    self.assertEqual(res[i], -105.283692410825)
                if name == "2_12_c_cf0":
                    self.assertEqual(res[i], -102.602803640785)
                if name == "10_5_c_cf0":
                    self.assertEqual(res[i], -115.729089277128)

            cleanup_dirs()
