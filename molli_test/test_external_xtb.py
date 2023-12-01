import unittest as ut

import numpy as np
import molli as ml

from joblib import delayed,Parallel
from molli.external import XTBDriver

# how to handle this?
# try:
#     from rdkit.Chem.PropertyMol import PropertyMol
# except:
#     _RDKIT_INSTALLED = False
# else:
#     _RDKIT_INSTALLED = True
# try:
#     from rdkit.Chem.Draw import IPythonConsole
# except:
#     _IPYTHON_INSTALLED = False
# else:
#     from molli.external import _rdkit
#     _IPYTHON_INSTALLED = True


class XTBTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    # @ut.skipUnless(_RDKIT_INSTALLED, "RDKit is not installed in current environment.")
    # @ut.skipUnless(_IPYTHON_INSTALLED, "IPython is not installed in current environment.")
    def test_xtb_optimize(self):
        # test with cinchonidine library
        mlib1 = ml.MoleculeLibrary(ml.files.cinchonidines)
        #Cinchonidine Charges = 1
        for m in mlib1:
            m.charge = 1

        # testing in serial works fine
        xtb = XTBDriver(nprocs=4)
        res = [xtb.optimize(m) for m in mlib1]
        for m1, m2 in zip(mlib1, res):
            self.assertNotAlmostEqual(np.linalg.norm(m1.coords - m2.coords), 0) # make sure the atom coordinates have moved


        # testing in parallel breaks it
        with self.assertRaises(AttributeError):
            xtb = XTBDriver(nprocs=4)
            res = Parallel(n_jobs=32, verbose=50)(
            delayed(xtb.optimize)(
                M=m, 
                method="gff",
                ) for m in mlib1)
            for m1, m2 in zip(mlib1, res):
                self.assertNotAlmostEqual(np.linalg.norm(m1.coords - m2.coords), 0) # make sure the atom coordinates have moved
        
        