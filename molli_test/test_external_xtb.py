import unittest as ut

import numpy as np
import molli as ml
import subprocess
import os
from pathlib import Path
import shutil

from joblib import delayed,Parallel
from molli.external.xtb import XTBDriver
from molli.config import BACKUP_DIR
from molli.config import SCRATCH_DIR


# a little clunky but will check if xTB is installed
try:
    out = subprocess.run(['xtb', '--version'], capture_output=True)
    if out.stderr == b'normal termination of xtb\n':
        _XTB_INSTALLED = True
except FileNotFoundError:
    _XTB_INSTALLED = False

_CURRENT_BACKUP_DIR = BACKUP_DIR
_CURRENT_SCRATCH_DIR = SCRATCH_DIR
_TEST_BACKUP_DIR: Path  = ml.config.HOME / "test_backup"
_TEST_SCRATCH_DIR: Path  = ml.config.HOME / "test_scratch"

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
    def test_xtb_optimize(self):

        source = ml.MoleculeLibrary(ml.files.cinchonidines)
        # test with cinchonidine library
        # with ml.MoleculeLibrary.reading(ml.files.cinchonidines) as source:
        with source.reading():
            prep_dirs()
            for m_name in source:
                m = source[m_name]
                m.charge = 1
            
            xtb = XTBDriver(nprocs=4)
            res = [xtb.optimize(source[m_name]) for m_name in source]

            for m1, m2 in zip([source[m_name] for m_name in source], res):
                self.assertEqual(m1.name, m2.name, 'Names must be the same!')
                self.assertNotAlmostEqual(np.linalg.norm(m1.coords - m2.coords), 0) # make sure the atom coordinates have moved

            cleanup_dirs()
            prep_dirs()
            # testing in parallel breaks it
            # with self.assertRaises(AttributeError):
            xtb = XTBDriver(nprocs=4)
            res = Parallel(n_jobs=32, verbose=50,prefer='threads')(
            delayed(xtb.optimize)(
                M=source[m_name], 
                method="gff",
                ) for m_name in source)
            for m1, m2 in zip([source[m_name] for m_name in source], res):
                self.assertEqual(m1.name, m2.name, 'Names must be the same!')
                self.assertNotAlmostEqual(np.linalg.norm(m1.coords - m2.coords), 0) # make sure the atom coordinates have moved
            cleanup_dirs()
        
    
    @ut.skipUnless(_XTB_INSTALLED, "xtb is not installed in current environment.")
    def test_xtb_energy(self):

        prep_dirs()

        # test with cinchonidine library
        mlib1 = ml.MoleculeLibrary(ml.files.cinchonidines)
        print(mlib1) # ALEX LOOK AT THIS! AN EMPTY COLLECTION. WHY DOES NOTHING FAIL??
        #Cinchonidine Charges = 1
        for m in mlib1:
            m.charge = 1

        # testing in serial works fine
        xtb = XTBDriver(nprocs=4)
        res = [xtb.energy(m) for m in mlib1]

        # we will spot check several of these based on output from separate call to xtb
        for i, energy in enumerate(res):
            if mlib1[i].name == '1_5_c':
                self.assertEqual(energy, -105.591624613587)
            if mlib1[i].name == '2_12_c':
                self.assertEqual(energy, -102.911928497077)
            if mlib1[i].name == '10_5_c':
                self.assertEqual(energy, -116.035733938867)

        cleanup_dirs()
        
        