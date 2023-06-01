import unittest as ut
import molli as ml


class PromoleculeTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_promolecule_empty_constructor(self):
        """Tests if empty promolecule is correctly created"""
        empty1 = ml.Promolecule()
        empty2 = ml.Promolecule()

        # Default tests here
        assert empty1.n_atoms == 0, "This should be an empty promolecule"
        self.assertEqual(
            empty1.molecular_weight,
            0.0,
            "Empty promolecule must have a molecular weight of 0.0",
        )

        # This looks kind of stupid, but there was a weird glitch in `molli 0.1.2` that was effectively doing this...
        # Turns out it was due to some_variable = [], which reused the empty list address (ugh)
        assert id(empty1._atoms) != id(
            empty2._atoms
        ), "The ID of atom lists are different"

    def test_promolecule_natoms_constructor(self):
        pm = ml.Promolecule(n_atoms=10)

    def test_promolecule_list_constructor(self):
        # This more or less just tests if all elements can end up in the promolecule upon such a creation
        pm = ml.Promolecule([elt for elt in ml.Element])

    def test_new_atom(self):
        """Tests programmatic creation of a simple water promolecule"""
        pm = ml.Promolecule()
        # This tests if string description of the element is a viable constructor option
        pm.append_atom(ml.Atom(ml.Element.H, label="H1"))
        pm.append_atom(ml.Atom("H", label="H2"))
        pm.append_atom(ml.Atom(8, label="O3"))

        self.assertEqual(
            pm.n_atoms, 3, "This promolecule (H2O) must have 3 atoms"
        )
        self.assertEqual(
            pm.molecular_weight,
            18.0150,
            "H2O promolecule molecular weight is not correct",
        )
        self.assertEqual(
            pm.formula,
            "H2 O1",
            "Formula looks a bit unusual, but it is easier to parse.",
        )

    def test_atoms_property(self):
        """Tests if the atoms property is programmatically protected"""
        pm = ml.Promolecule()

        self.assertIsInstance(pm.atoms, list)
        with self.assertRaises(AttributeError):
            pm.atoms = [
                1,
                2,
                3,
            ]  # This should fail as the property is protected

    def test_atom_indexing(self):
        """Tests atom indexing procedures and their stability with respect to atom creation/deletion/rearrangement"""
        pm = ml.Promolecule()
        # old deprecated syntax
        # h1 = ml.Atom.add_to(pm, ml.Element.H, label="H1")
        # h2 = ml.Atom.add_to(pm, ml.Element.H, label="H2")
        # o3 = ml.Atom.add_to(pm, ml.Element.O, label="O3")

        h1 = ml.Atom(ml.Element.H, label="H1")
        h2 = ml.Atom(ml.Element.H, label="H2")
        o3 = ml.Atom(ml.Element.O, label="O3")

        pm = ml.Promolecule([h1, h2, o3])

        # Simple tests of atom indexing
        self.assertEqual(
            pm.index_atom(o3),
            2,
            "That would be the third atom in that promolecule",
        )

        # Atom deletion tests
        pm.del_atom(h2)
        self.assertEqual(
            pm.index_atom(o3),
            1,
            (
                "That would now be the second atom in that promolecule, since"
                " h2 was deleted"
            ),
        )

        with self.assertRaises(ValueError):
            pm.index_atom(
                h2
            )  # This should fail as h2 is no longer in the atom list

        h2 = ml.Atom(ml.Element.H, label="H2")
        pm.append_atom(h2)
        hydrogens = list(pm.yield_atoms_by_element("H"))

        self.assertListEqual(
            hydrogens, [h1, h2], "This is the correct list of hydrogens"
        )
