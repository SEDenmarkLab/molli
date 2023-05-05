import unittest as ut

from molli import chem, files, Molecule
import numpy as np
from pathlib import Path


class MoleculeTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_yield_from_mol2(self):
        with open(files.pentane_confs_mol2) as f:
            lst_structs: list[chem.Molecule] = list(
                chem.Molecule.yield_from_mol2(f, name="pentane")
            )

        self.assertEqual(len(lst_structs), 7)

        for m in lst_structs:
            self.assertEqual(m.n_atoms, 17)
            self.assertEqual(m.n_bonds, 16)

    # @ut.skip("Not implemented yet")
    def test_molecule_cloning(self):
        with open(files.pentane_confs_mol2) as f:
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

    def test_dump_mol2(self):
        for file in [
            files.pentane_confs_mol2,
            files.dendrobine_mol2,
            files.dummy_mol2,
            files.fxyl_mol2,
            files.nanotube_mol2
        ]:

            with open(file) as f:
                m1 = chem.Molecule.load_mol2(f)
            
            new_m1 = next(Molecule.yield_from_mol2(Molecule.dumps_mol2(m1)))

            np.testing.assert_array_equal(m1._coords,new_m1._coords)

            self.assertListEqual([a1.element.symbol for a1 in m1.atoms],[new_a1.element.symbol for new_a1 in new_m1.atoms])

            self.assertListEqual([b1.btype for b1 in m1.bonds],[new_b1.btype for new_b1 in new_m1.bonds])