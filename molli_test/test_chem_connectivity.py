import unittest as ut

from molli import chem


class ConnectivityTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_connectivity_empty_constructor(self):
        """Tests if empty promolecule is correctly created"""
        empty1 = chem.Connectivity()
        empty2 = chem.Connectivity()

        # Default tests here
        assert empty1.n_bonds == 0, "This should be an empty connectivity table"
        assert id(empty1._bonds) != id(empty2._bonds)

    def test_connectivity_bond_creation(self):

        ###############################################
        # This creates a water molecule from scratch
        cn = chem.Connectivity()

        h1 = chem.Atom.add_to(cn, chem.Element.H, label="H1")
        h2 = chem.Atom.add_to(cn, "H", label="H2")
        o3 = chem.Atom.add_to(cn, chem.Element.O, label="O3")

        b1 = chem.Bond.add_to(cn, h1, o3)
        b2 = chem.Bond.add_to(cn, h2, o3, order=1, stereo=True)
        ###############################################

        self.assertEqual(cn.n_atoms, 3, "H2O has 3 atoms")
        self.assertEqual(cn.n_bonds, 2, "H2O has 2 bonds")

        self.assertIsNone(cn.lookup_bond(h1, h2), "H2O has no bond between hydrogens")
        self.assertEqual(cn.lookup_bond(o3, h2), b2)

        self.assertSetEqual(
            set(cn.bonds_with_atom(o3)), {b1, b2}, "Bonds with oxygen atom O3"
        )
        self.assertEqual(cn.bonded_valence(h1), 1)
        self.assertEqual(cn.bonded_valence(h2), 1)
        self.assertEqual(cn.bonded_valence(o3), 2)

        self.assertEqual(h1, b1 % o3)
