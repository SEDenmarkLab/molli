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
import importlib.util


# a little clunky but will check if xTB is installed
try:
    out = subprocess.run(["xtb", "--version"], capture_output=True)
    if out.stderr == b"normal termination of xtb\n":
        _XTB_INSTALLED = True
except FileNotFoundError:
    _XTB_INSTALLED = False


def is_package_installed(pkg_name):
    return importlib.util.find_spec(pkg_name) is not None


_CURRENT_BACKUP_DIR = BACKUP_DIR
_CURRENT_SCRATCH_DIR = SCRATCH_DIR
_TEST_BACKUP_DIR: Path = ml.config.HOME / "test_backup"
_TEST_SCRATCH_DIR: Path = ml.config.HOME / "test_scratch"
_TEST_LOG_DIR: Path = ml.config.HOME / "test_log"
_TEST_ERROR_DIR: Path = ml.config.HOME / "test_error"


def prep_dirs():
    ml.config.BACKUP_DIR = _TEST_BACKUP_DIR
    ml.config.SCRATCH_DIR = _TEST_SCRATCH_DIR
    # _TEST_BACKUP_DIR.mkdir()
    # _TEST_SCRATCH_DIR.mkdir()


def cleanup_dirs():
    shutil.rmtree(_TEST_BACKUP_DIR)
    shutil.rmtree(_TEST_SCRATCH_DIR)
    shutil.rmtree(_TEST_LOG_DIR)
    shutil.rmtree(_TEST_ERROR_DIR)
    ml.config.BACKUP_DIR = _CURRENT_BACKUP_DIR
    ml.config.SCRATCH_DIR = _CURRENT_SCRATCH_DIR


class XTBTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    @ut.skipUnless(_XTB_INSTALLED, "xtb is not installed in current environment.")
    def test_xtb_optimize_parallel(self):
        xtb = XTBDriver(nprocs=4)

        source = ml.MoleculeLibrary(ml.files.cinchonidine_no_conf)

        target = ml.MoleculeLibrary(
            ml.config.SCRATCH_DIR / "dest.mlib",
            overwrite=True,
            readonly=False,
            comment="We did it!",
        )

        prep_dirs()

        ml.pipeline.jobmap(
            xtb.optimize_m,
            source,
            target,
            cache_dir=ml.config.BACKUP_DIR,
            error_dir=_TEST_ERROR_DIR,
            log_dir=_TEST_LOG_DIR,
            scratch_dir=ml.config.SCRATCH_DIR,
            scheduler="local",
            n_workers=4,
            n_jobs_per_worker=8,
            verbose=True,
            progress=True,
        )

        res = ml.MoleculeLibrary(ml.config.SCRATCH_DIR / "dest.mlib")

        with res.reading(), source.reading():
            for name in res:
                m1, m2 = source[name], res[name]
                self.assertNotAlmostEqual(np.linalg.norm(m1.coords - m2.coords), 0)
            cleanup_dirs()

    @ut.skipUnless(_XTB_INSTALLED, "xtb is not installed in current environment.")
    def test_xtb_energy_parallel(self):
        xtb = XTBDriver(nprocs=4)

        source = ml.MoleculeLibrary(ml.files.cinchonidine_no_conf)

        target = ml.MoleculeLibrary(
            ml.config.SCRATCH_DIR / "dest.mlib",
            overwrite=True,
            readonly=False,
            comment="We did it!",
        )

        prep_dirs()

        ml.pipeline.jobmap(
            xtb.energy,
            source,
            target,
            cache_dir=ml.config.BACKUP_DIR,
            error_dir=_TEST_ERROR_DIR,
            log_dir=_TEST_LOG_DIR,
            scratch_dir=ml.config.SCRATCH_DIR,
            scheduler="local",
            n_workers=4,
            n_jobs_per_worker=8,
            verbose=True,
            progress=True,
        )

        res = ml.MoleculeLibrary(ml.config.SCRATCH_DIR / "dest.mlib")

        with res.reading():
            for name in res:
                m = res[name]
                if name == "1_5_c_cf0":
                    self.assertEqual(m.attrib["energy"], -105.283692410825)
                if name == "2_12_c_cf0":
                    self.assertEqual(m.attrib["energy"], -102.602803640785)
                if name == "10_5_c_cf0":
                    self.assertEqual(m.attrib["energy"], -115.729089277128)
            cleanup_dirs()

    @ut.skipUnless(is_package_installed("pandas"), "pandas is not installed")
    def test_atom_properties_parallel(self):
        xtb = XTBDriver(nprocs=4)

        source = ml.MoleculeLibrary(ml.files.cinchonidine_no_conf)

        target = ml.MoleculeLibrary(
            ml.config.SCRATCH_DIR / "dest.mlib",
            overwrite=True,
            readonly=False,
            comment="We did it!",
        )

        prep_dirs()

        ml.pipeline.jobmap(
            xtb.atom_properties_m,
            source,
            target,
            cache_dir=ml.config.BACKUP_DIR,
            error_dir=_TEST_ERROR_DIR,
            log_dir=_TEST_LOG_DIR,
            scratch_dir=ml.config.SCRATCH_DIR,
            scheduler="local",
            n_workers=4,
            n_jobs_per_worker=8,
            verbose=True,
            progress=True,
        )

        res = ml.MoleculeLibrary(ml.config.SCRATCH_DIR / "dest.mlib")

        with res.reading():
            for name in res:
                m = res[name]
                a = m.get_atom(0)
                for att in a.attrib:
                    attribs = [a.attrib[att] for a in m.atoms]
                    self.assertEqual(len(m.atoms), len(attribs))
                    self.assertTrue(attribs)
            cleanup_dirs()
