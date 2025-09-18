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
    if b'normal termination' in out.stderr:
        _XTB_INSTALLED = True
except FileNotFoundError:
    _XTB_INSTALLED = False

def is_package_installed(pkg_name):
    return importlib.util.find_spec(pkg_name) is not None


# def cleanup_dirs(backup_dir, scratch_dir):
#     shutil.rmtree(backup_dir)
#     shutil.rmtree(scratch_dir)
    # for file in (SCRATCH_DIR).glob("*.mlib"):
    #     file.unlink()

class XTBTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""
    
    @ut.skipUnless(_XTB_INSTALLED, "xtb is not installed in current environment.")
    def test_xtb_optimize_parallel(self):
        xtb = XTBDriver(nprocs=4)

        _TEST_DIR = SCRATCH_DIR / 'xtb_opt'
        _TEST_BACKUP_DIR = _TEST_DIR / 'backup'
        _TEST_SCRATCH_DIR = _TEST_DIR / 'scratch'

        [x.mkdir(parents=True, exist_ok=True) for x in [_TEST_DIR, _TEST_BACKUP_DIR, _TEST_SCRATCH_DIR]]

        init_lib = ml.MoleculeLibrary(ml.files.cinchonidine_no_conf)

        source = ml.MoleculeLibrary(_TEST_DIR / 'small.mlib', readonly=False, overwrite=True)

        with init_lib.reading(), source.writing():
            for i, k in enumerate(init_lib):
                source[k] = init_lib[k]
                if i == 10:
                    break

        target = ml.MoleculeLibrary(
            _TEST_DIR / "dest.mlib",
            overwrite=True,
            readonly=False,
            comment="We did it!",
        )

        ml.pipeline.jobmap(
            xtb.optimize_m,
            source,
            target,
            cache_dir=_TEST_BACKUP_DIR,
            scratch_dir=_TEST_SCRATCH_DIR,
            n_workers=2,
            verbose=True,
            progress=True
        )


        with target.reading(), source.reading():
            self.assertTrue(target.n_items == source.n_items)
            for name in target:
                m1 = source[name]
                m2 = target[name]
                self.assertNotAlmostEqual(np.linalg.norm(m1.coords - m2.coords), 0)
        
        shutil.rmtree(_TEST_DIR)

    @ut.skipUnless(_XTB_INSTALLED, "xtb is not installed in current environment.")
    def test_xtb_energy_parallel(self):
        xtb = XTBDriver(nprocs=4,memory=4000)

        _TEST_DIR = SCRATCH_DIR / 'xtb_energy'
        _TEST_BACKUP_DIR = _TEST_DIR / 'backup'
        _TEST_SCRATCH_DIR = _TEST_DIR / 'scratch'

        [x.mkdir(parents=True, exist_ok=True) for x in [_TEST_DIR, _TEST_BACKUP_DIR, _TEST_SCRATCH_DIR]]

        init_lib = ml.MoleculeLibrary(ml.files.cinchonidine_no_conf)

        source = ml.MoleculeLibrary(_TEST_DIR / 'small.mlib', readonly=False, overwrite=True)

        with init_lib.reading(), source.writing():
            for i, k in enumerate(init_lib):
                source[k] = init_lib[k]
                if i == 10:
                    break

        target = ml.MoleculeLibrary(
            _TEST_DIR / "dest.mlib",
            overwrite=True,
            readonly=False,
            comment="We did it!",
        )

        ml.pipeline.jobmap(
            xtb.energy,
            source,
            target,
            cache_dir=_TEST_BACKUP_DIR.as_posix(),
            scratch_dir=_TEST_SCRATCH_DIR.as_posix(),
            n_workers=2,
            verbose=True,
            progress=True,
        )

        with target.reading(), source.reading():
            self.assertTrue(target.n_items == source.n_items)
            for name in target:
                m = target[name]
                self.assertTrue('energy' in m.attrib)
        
        shutil.rmtree(_TEST_DIR)

    @ut.skipUnless(is_package_installed("pandas"), "pandas is not installed")
    @ut.skipUnless(_XTB_INSTALLED, "xtb is not installed in current environment.")
    def test_atom_properties_parallel(self):
        xtb = XTBDriver(nprocs=4)
        
        _TEST_DIR = SCRATCH_DIR / 'xtb_atom'
        _TEST_BACKUP_DIR = _TEST_DIR / 'backup'
        _TEST_SCRATCH_DIR = _TEST_DIR / 'scratch'

        [x.mkdir(parents=True, exist_ok=True) for x in [_TEST_DIR, _TEST_BACKUP_DIR, _TEST_SCRATCH_DIR]]

        init_lib = ml.MoleculeLibrary(ml.files.cinchonidine_no_conf)

        source = ml.MoleculeLibrary(_TEST_DIR / 'small.mlib', readonly=False, overwrite=True)

        with init_lib.reading(), source.writing():
            for i, k in enumerate(init_lib):
                source[k] = init_lib[k]
                if i == 10:
                    break

        target = ml.MoleculeLibrary(
            _TEST_DIR / "dest.mlib",
            overwrite=True,
            readonly=False,
            comment="We did it!",
        )

        ml.pipeline.jobmap(
            xtb.atom_properties_m,
            source,
            target,
            cache_dir=_TEST_BACKUP_DIR.as_posix(),
            scratch_dir=_TEST_SCRATCH_DIR.as_posix(),
            n_workers=2,
            verbose=True,
            progress=True,
        )


        with target.reading(), source.reading():
            self.assertTrue(target.n_items == source.n_items)
            for name in target:
                m = target[name]
                a = m.get_atom(0)
                for att in a.attrib:
                    self.assertTrue(a.attrib[att] != '')
                attribs = [a.attrib[att] for a in m.atoms]
                self.assertEqual(len(m.atoms), len(attribs))

        shutil.rmtree(_TEST_DIR)
