# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Alexander S. Shved <shvedalx@illinois.edu>
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
Testing the `Promolecule` class functionality
"""

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
        """Tests constructing promolecule with n_atoms"""
        pm = ml.Promolecule(n_atoms=10)
        self.assertEqual(len(pm._atoms), 10)

        pm = ml.Promolecule(n_atoms=0)
        self.assertEqual(len(pm._atoms), 0)

        # can't instantiate with negative atoms
        with self.assertRaises(ValueError):
            pm = ml.Promolecule(n_atoms=-1)

    def test_promolecule_list_constructor(self):
        """Tests constructing promolecule from lists of ElementLike or Atoms"""
        # This more or less just tests if all elements can end up in the promolecule upon such a creation
        pm = ml.Promolecule([elt for elt in ml.Element])
        self.assertEqual(len(pm._atoms), len(ml.Element))

        # make sure it works from strings
        pm = ml.Promolecule([repr(elt) for elt in ml.Element])
        self.assertEqual(len(pm._atoms), len(ml.Element))

        # make sure it works from ml.Atom objects
        pm = ml.Promolecule([ml.Atom(elt) for elt in ml.Element])
        self.assertEqual(len(pm._atoms), len(ml.Element))

    def test_promolecule_cloning(self):
        """Tests constructing promolecule from promolecule, i.e. cloning"""
        pm1 = ml.Promolecule([elt for elt in ml.Element])
        pm2 = ml.Promolecule(pm1)

        self.assertNotEqual(id(pm1), id(pm2))
        self.assertNotEqual(id(pm1._atoms), id(pm2._atoms))

        for a1, a2 in zip(pm1._atoms, pm2._atoms):
            self.assertNotEqual(a1, a2)

        for a1, a2 in zip(pm1._atoms, pm2._atoms):
            self.assertNotEqual(id(a1), id(a2))

    def test_new_atom(self):
        """Tests programmatic creation of a simple water promolecule"""
        pm = ml.Promolecule()
        # This tests if string description of the element is a viable constructor option
        pm.append_atom(ml.Atom(ml.Element.H, label="H1"))
        pm.append_atom(ml.Atom("H", label="H2"))
        pm.append_atom(ml.Atom(8, label="O3"))

        self.assertEqual(pm.n_atoms, 3, "This promolecule (H2O) must have 3 atoms")
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

    @ut.skip(
        "This test will likely be removed in the future."
        "Naming conventions are about to become more relaxed."
    )
    def test_name_property(self):
        """Tests the names property setter, with user warning"""
        pm = ml.Promolecule()

        self.assertEqual(pm.name, "unknown")
        # the " " is not allowed in a name, should be replaced with "_"
        with self.assertWarns(UserWarning):
            pm.name = "awesome promolecule"

        self.assertEqual(pm.name, "awesome_promolecule")

    def test_get_atom(self):
        """Tests get atom functionality"""
        pm = ml.Promolecule()
        # This tests if string description of the element is a viable constructor option
        pm.append_atom(ml.Atom(ml.Element.H, label="H1"))
        pm.append_atom(ml.Atom("H", label="H2"))
        oxygenatom = ml.Atom(8, label="O3")
        pm.append_atom(oxygenatom)

        # get as atom object
        got = pm.get_atom(oxygenatom)
        self.assertEqual(id(got), id(oxygenatom))

        # get as atom index
        got = pm.get_atom(2)
        self.assertEqual(id(got), id(oxygenatom))

        # get as atom label
        got = pm.get_atom("O3")
        self.assertEqual(id(got), id(oxygenatom))

        # get as element
        got = pm.get_atom(ml.Element.O)
        self.assertEqual(id(got), id(oxygenatom))

        # get some nonsense, should raise error
        with self.assertRaises(ValueError):
            got = pm.get_atom(ml.Promolecule())  # promolecule
            got = pm.get_atom(3)  # index out of range
            got = pm.get_atom(None)  # No input

    def test_get_atoms(self):
        """Tests get atoms functionality"""
        pm = ml.Promolecule()
        # This tests if string description of the element is a viable constructor option
        h1 = ml.Atom(ml.Element.H, label="H1")
        h2 = ml.Atom("H", label="H2")
        pm.append_atom(h1)
        pm.append_atom(h2)
        pm.append_atom(ml.Atom(8, label="O3"))

        got = pm.get_atoms(0, 1)
        self.assertEqual(id(got[0]), id(h1))
        self.assertEqual(id(got[1]), id(h2))

    def test_get_atom_index(self):
        """Tests get atom index functionality"""
        pm = ml.Promolecule()
        # This tests if string description of the element is a viable constructor option
        pm.append_atom(ml.Atom(ml.Element.H, label="H1"))
        pm.append_atom(ml.Atom("H", label="H2"))
        oxygenatom = ml.Atom(8, label="O3")
        pm.append_atom(oxygenatom)

        # get as atom object
        got = pm.get_atom_index(oxygenatom)
        self.assertEqual(got, 2)

        # get as atom index... why does this functionality exist?
        got = pm.get_atom_index(2)
        self.assertEqual(got, 2)
        with self.assertRaises(ValueError):
            got = pm.get_atom_index(3)

        # get as atom label
        got = pm.get_atom_index("O3")
        self.assertEqual(got, 2)

        # get some nonsense, should raise error
        with self.assertRaises(ValueError):
            got = pm.get_atom_index(ml.Promolecule())  # promolecule
            got = pm.get_atom_index(3)  # index out of range
            got = pm.get_atom_index(None)  # No input

    def test_get_atom_indices(self):
        """Tests get atom indices functionality"""
        pm = ml.Promolecule()
        # This tests if string description of the element is a viable constructor option
        h1 = ml.Atom(ml.Element.H, label="H1")
        h2 = ml.Atom("H", label="H2")
        pm.append_atom(h1)
        pm.append_atom(h2)
        pm.append_atom(ml.Atom(8, label="O3"))

        got = pm.get_atom_indices(h1, h2)
        self.assertEqual(got[0], 0)
        self.assertEqual(got[1], 1)

    def test_del_atom1(self):
        """Test deleting atom functionality"""

        pm = ml.Promolecule()
        pm.append_atom(ml.Atom(ml.Element.H, label="H1"))
        pm.append_atom(ml.Atom("H", label="H2"))
        pm.append_atom(ml.Atom(8, label="O3"))

        natoms1 = pm.n_atoms

        oxygenatom = pm.get_atom("O3")
        pm.del_atom(oxygenatom)

        self.assertEqual(natoms1 - 1, pm.n_atoms)
        with self.assertRaises(StopIteration):
            pm.get_atom("O3")
            pm.get_atom(2)

    def test_del_atom2(self):
        """More atom deletion testing"""
        pm = ml.Promolecule(ml.load(ml.files.dendrobine_mol2))
        pm.del_atom(0)

    def test_append_atoms(self):
        """Test appending atom functionality"""

        pm = ml.Promolecule()
        self.assertEqual(pm.n_atoms, 0)
        pm.append_atom(ml.Atom(ml.Element.H, label="H1"))
        self.assertEqual(pm.n_atoms, 1)
        pm.append_atom(ml.Atom("H", label="H2"))
        self.assertEqual(pm.n_atoms, 2)
        pm.append_atom(ml.Atom(8, label="O3"))
        self.assertEqual(pm.n_atoms, 3)

    def test_index_atom(self):
        """Tests atom indexing procedures and their stability with respect to atom creation/deletion/rearrangement"""
        pm = ml.Promolecule()

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

        self.assertEqual(o3.idx, 2, "This tests the alternative way of indexing atoms")

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
            pm.index_atom(h2)  # This should fail as h2 is no longer in the atom list

        h2 = ml.Atom(ml.Element.H, label="H2")
        pm.append_atom(h2)
        hydrogens = list(pm.yield_atoms_by_element("H"))

        self.assertListEqual(
            hydrogens, [h1, h2], "This is the correct list of hydrogens"
        )

    def test_yield_atom_by_element(self):
        """Tests yield atom by element functionality"""
        pm = ml.Promolecule()

        h1 = ml.Atom(ml.Element.H, label="H1")
        h2 = ml.Atom(ml.Element.H, label="H2")
        o3 = ml.Atom(ml.Element.O, label="O3")

        pm = ml.Promolecule([h1, h2, o3])

        hs = pm.yield_atoms_by_element("H")

        self.assertEqual(id(next(hs)), id(h1))
        self.assertEqual(id(next(hs)), id(h2))
        with self.assertRaises(StopIteration):
            next(hs)

        o = pm.yield_atoms_by_element(8)
        self.assertEqual(id(next(o)), id(o3))
        with self.assertRaises(StopIteration):
            next(o)

    def test_yield_attachment_points(self):
        """Tests yield attachment point functionality"""
        pm = ml.Promolecule()

        ap1 = ml.Atom(atype=ml.AtomType.AttachmentPoint, label="ap1")
        ap2 = ml.Atom(atype=ml.AtomType.AttachmentPoint, label="ap2")
        o3 = ml.Atom(ml.Element.O, label="O3")

        pm = ml.Promolecule([ap1, ap2, o3])

        hs = pm.yield_attachment_points()

        self.assertEqual(id(next(hs)), id(ap1))
        self.assertEqual(id(next(hs)), id(ap2))
        with self.assertRaises(StopIteration):
            next(hs)

        self.assertEqual(len(pm.get_attachment_points()), 2)

    def test_yield_atom_by_label(self):
        """Tests yield atom by label functionality"""
        pm = ml.Promolecule()

        h1 = ml.Atom(ml.Element.H, label="H1")
        h2 = ml.Atom(ml.Element.H, label="H2")
        o3 = ml.Atom(ml.Element.O, label="O3")

        pm = ml.Promolecule([h1, h2, o3])

        os = pm.yield_atoms_by_label("O3")

        self.assertEqual(id(next(os)), id(o3))
        with self.assertRaises(StopIteration):
            next(os)

    @ut.skip("promolecule.sort_atoms has not been implemented.")
    def test_sort_atoms(self):
        return None

    def test_pickle_serialization(self):
        import pickle

        a1 = ml.Atom("H")
        a_pkl = pickle.dumps(a1)
        a2 = pickle.loads(a_pkl)

        self.assertNotEqual(a1, a2)
        self.assertEqual(a1.element, a2.element)

        pm1 = ml.Promolecule([1, 8, 1])  # water promolecule
        pm_pkl = pickle.dumps(pm1)
        pm2 = pickle.loads(pm_pkl)
