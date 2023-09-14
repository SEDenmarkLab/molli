import unittest as ut

import molli as ml
import numpy as np
from copy import deepcopy

try:
    from openbabel import openbabel as ob
except:
    _OPENBABEL_INSTALLED = False
else:
    from molli.external import openbabel as _ob
    _OPENBABEL_INSTALLED = True

class OpenbabelTC(ut.TestCase):

    @ut.skipUnless(_OPENBABEL_INSTALLED, "Openbabel is not installed in current environment")
    def test_ob_mol2(self):
        with ml.files.dendrobine_mol2.open() as f:
            mlmol = ml.Molecule.load_mol2(f)
        
        mol2_block = _ob.to_mol2_w_ob(mlmol)

        new_mlmol = next(ml.Structure.yield_from_mol2(mol2_block))

        self.assertEqual(len(mlmol.atoms),len(new_mlmol.atoms))
        self.assertEqual(len(mlmol.bonds),len(new_mlmol.bonds))
    
    @ut.skipUnless(_OPENBABEL_INSTALLED, "Openbabel is not installed in current environment")
    def test_ob_opt(self):
        with ml.files.dendrobine_mol2.open() as f:
            mlmol = next(ml.Molecule.yield_from_mol2(f))

        old_coords = deepcopy(mlmol.coords)

        opt_mlmol = _ob.obabel_optimize(mlmol,ff='UFF')

        opt_coords = opt_mlmol.coords
        self.assertTupleEqual(old_coords.shape, opt_coords.shape)

        np.testing.assert_raises(AssertionError,np.testing.assert_array_equal, old_coords,opt_coords)
