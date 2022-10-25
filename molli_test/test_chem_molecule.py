import unittest as ut

from molli import chem
import numpy as np
from pathlib import Path


class MoleculeTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_yield_from_mol2(self):
        fpath = Path(__file__).parent / "files" / "pentane_confs.mol2"

        with open(fpath, "rt") as f:
            lst_structs: list[chem.Molecule] = list(
                chem.Molecule.yield_from_mol2(f, name="pentane")
            )

        self.assertEqual(len(lst_structs), 7)

        for m in lst_structs:
            self.assertEqual(m.n_atoms, 17)
            self.assertEqual(m.n_bonds, 16)

    @ut.skip("Not implemented yet")
    def test_molecule_cloning(self):
        with open(Path(__file__).parent / "files" / "pentane_confs.mol2", "rt") as f:
            m1 = chem.Molecule.from_mol2(f, name="pentane")

        m2 = chem.Molecule(m1)

        self.assertNotEqual(id(m1._atoms), id(m2._atoms))
        self.assertNotEqual(id(m1._bonds), id(m2._bonds))
        self.assertNotEqual(id(m1._coords), id(m2._coords))

        self.assertEqual(np.linalg.norm(m1.coords - m2.coords), 0)
