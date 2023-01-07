import unittest as ut

from molli import chem, files
import numpy as np
from pathlib import Path


class MoleculeTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_yield_from_mol2(self):
        with files.mol2.pentane_confs.open("t") as f:
            lst_structs: list[chem.Molecule] = list(
                chem.Molecule.yield_from_mol2(f, name="pentane")
            )

        self.assertEqual(len(lst_structs), 7)

        for m in lst_structs:
            self.assertEqual(m.n_atoms, 17)
            self.assertEqual(m.n_bonds, 16)

    # @ut.skip("Not implemented yet")
    def test_molecule_cloning(self):
        with files.mol2.pentane_confs.open("t") as f:
            m1 = chem.Molecule.load_mol2(f, name="pentane")

        m2 = chem.Molecule(m1)

        self.assertNotEqual(id(m1._atoms), id(m2._atoms))
        self.assertNotEqual(id(m1._bonds), id(m2._bonds))

        for a1, a2 in zip(m1.atoms, m2.atoms):
            self.assertNotEqual(a1, a2)

        for b1, b2 in zip(m1.bonds, m2.bonds):
            self.assertNotEqual(b1, b2)

        self.assertNotEqual(id(m1._coords), id(m2._coords))

        self.assertEqual(np.linalg.norm(m1.coords - m2.coords), 0)
