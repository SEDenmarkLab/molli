import unittest as ut

import molli as ml


class ConnectivityTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_connectivity_empty_constructor(self):
        """Tests if empty promolecule is correctly created"""
        empty1 = ml.Connectivity()
        empty2 = ml.Connectivity()

        # Default tests here
        assert empty1.n_bonds == 0, "This should be an empty connectivity table"
        assert id(empty1._bonds) != id(empty2._bonds)

    def test_connectivity_bond_creation(self):

        ###############################################
        # This creates a water molecule from scratch

        h1 = ml.Atom(ml.Element.H, label="H1")
        h2 = ml.Atom("H", label="H2")
        o3 = ml.Atom(ml.Element.O, label="O3")

        cn = ml.Connectivity([h1, h2, o3])

        b1 = ml.Bond(h1, o3, btype=ml.BondType.Single)
        b2 = ml.Bond(h2, o3, btype=ml.BondType.Single)

        cn.append_bonds(b1, b2)
        ###############################################

        self.assertEqual(cn.n_atoms, 3, "H2O must have 3 atoms")
        self.assertEqual(cn.n_bonds, 2, "H2O must have 2 bonds")

        self.assertIsNone(
            cn.lookup_bond(h1, h2), "H2O must have no bond between hydrogens"
        )
        self.assertEqual(cn.lookup_bond(o3, h2), b2)

        self.assertSetEqual(
            set(cn.bonds_with_atom(o3)), {b1, b2}, "Bonds with oxygen atom O3"
        )
        self.assertEqual(cn.bonded_valence(h1), 1.0)
        self.assertEqual(cn.bonded_valence(h2), 1.0)
        self.assertEqual(cn.bonded_valence(o3), 2.0)

        self.assertEqual(h1, b1 % o3)
