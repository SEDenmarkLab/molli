import unittest as ut

import molli as ml
from molli.external import _rdkit
import numpy as np
from rdkit.Chem.PropertyMol import PropertyMol


class RDKitTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_create_rdkit_mol(self):
        with ml.files.zincdb_fda_mol2.open() as f:
            structs = ml.Structure.yield_from_mol2(f)
            for s in structs:
                molli_mol = ml.Molecule(s,name=s.name)
                molli_mol_rdmol_dict = _rdkit.create_rdkit_mol(molli_mol)
                self.assertIsInstance(molli_mol_rdmol_dict[molli_mol], PropertyMol)

    def test_molli_mol_reorder(self):
        with ml.files.zincdb_fda_mol2.open() as f:
            structs = ml.Structure.yield_from_mol2(f)
            for s in structs:
                molli_mol = ml.Molecule(s,name=s.name)
                molli_mol_rdmol_dict = _rdkit.create_rdkit_mol(molli_mol)
                rd_can_mol, atom_reorder, bond_reorder = _rdkit.can_mol_order(molli_mol_rdmol_dict[molli_mol])
                molli_can_rdkit_dict = _rdkit.reorder_molecule(molli_mol,can_rdkit_mol_w_h=rd_can_mol, can_atom_reorder=atom_reorder, can_bond_reorder=bond_reorder)
                for molli_mol,rdkit_mol in molli_can_rdkit_dict.items():
                    can_rdkit_atom_elem = np.array([x.GetSymbol() for x in rdkit_mol.GetAtoms()])
                    new_molli_elem = np.array([atom.element.symbol for atom in molli_mol.atoms])
                    np.testing.assert_array_equal(can_rdkit_atom_elem,new_molli_elem)

    def test_rdkit_atom_filter(self):
        m1 = ml.Molecule.load_mol2(ml.files.dendrobine_mol2,name='dendrobine')

        molli_rdkit_dict = _rdkit.create_rdkit_mol(m1)
        af_mol = _rdkit.rdkit_atom_filter(molli_rdkit_dict[m1])

        af_mol_sp2_bool = af_mol.sp2_type()
        num_sp2_atoms = np.count_nonzero(af_mol_sp2_bool)
        self.assertEqual(num_sp2_atoms, 3)

        af_mol_het_neighbors_2 = af_mol.het_neighbors_2()
        num_het_neighbors_2 = np.count_nonzero(af_mol_het_neighbors_2)
        self.assertEqual(num_het_neighbors_2, 1)

        af_mol_dual_bool = af_mol.sp2_type() & af_mol.het_neighbors_2()
        num_af_mol_dual_bool = np.count_nonzero(af_mol_dual_bool)
        self.assertEqual(num_af_mol_dual_bool, 1)