import unittest as ut
import molli as ml
import importlib
import numpy
from numpy.testing import assert_array_almost_equal


def is_package_installed(pkg_name):
    return importlib.util.find_spec(pkg_name) is not None


class ReadWriteTC(ut.TestCase):

    def test_load(self):
        # Dendrobine molecule
        mol1 = ml.load(ml.files.dendrobine_mol2)
        mol2 = ml.load(ml.files.dendrobine_xyz)

        self.assertListEqual(mol1.elements, mol2.elements)
        assert_array_almost_equal(mol1.coords, mol2.coords, 4)

        # Pentane conformers
        ens1 = ml.load(ml.files.pentane_confs_mol2, otype="ensemble")
        ens2 = ml.load(ml.files.pentane_confs_xyz, otype="ensemble")
        self.assertEqual(ens1.n_conformers, 7)
        self.assertEqual(ens2.n_conformers, 7)
        assert_array_almost_equal(ens1.coords, ens2.coords, 4)

    @ut.skipIf(not is_package_installed("openbabel"), "Openbabel is not installed")
    def test_load_ob(self):
        # Dendrobine molecule
        mol1 = ml.load(ml.files.dendrobine_mol2)
        mol2 = ml.load(ml.files.dendrobine_xyz)
        mol3 = ml.load(ml.files.dendrobine_molv3, parser="obabel")

        self.assertListEqual(mol1.elements, mol2.elements)
        assert_array_almost_equal(mol1.coords, mol2.coords, 4)

        self.assertListEqual(mol1.elements, mol3.elements)
        assert_array_almost_equal(mol1.coords, mol3.coords, 4)

        # Pentane conformers
        ens1 = ml.load(ml.files.pentane_confs_mol2, otype="ensemble")
        ens2 = ml.load(
            ml.files.pentane_confs_molv3, otype="ensemble", parser="openbabel"
        )
        assert_array_almost_equal(ens1.coords, ens2.coords, 4)
        self.assertEqual(ens1.n_conformers, 7)

    def test_loads(self):
        mol1 = ml.loads(ml.files.dendrobine_mol2.read_text(), "mol2")
        mol2 = ml.loads(ml.files.dendrobine_xyz.read_text(), "xyz")

        self.assertListEqual(mol1.elements, mol2.elements)
        assert_array_almost_equal(mol1.coords, mol2.coords, 4)

    @ut.skipIf(not is_package_installed("openbabel"), "Openbabel is not installed")
    def test_loads_ob(self):
        mol1 = ml.loads(ml.files.dendrobine_mol2.read_text(), "mol2")
        mol2 = ml.loads(ml.files.dendrobine_xyz.read_text(), "xyz")
        mol3 = ml.loads(ml.files.dendrobine_molv3.read_text(), "mol", parser="obabel")

        self.assertListEqual(mol1.elements, mol2.elements)
        assert_array_almost_equal(mol1.coords, mol2.coords, 4)

        self.assertListEqual(mol1.elements, mol3.elements)
        assert_array_almost_equal(mol1.coords, mol3.coords, 4)

    def test_load_all(self):
        ml.load_all(ml.files.zincdb_fda_mol2, parser="molli")

    @ut.skipIf(not is_package_installed("openbabel"), "Openbabel is not installed")
    def test_load_all_ob(self):
        ml.load_all(ml.files.zincdb_fda_mol2, parser="molli")
        ml.load_all(ml.files.zincdb_fda_molv3, parser="openbabel")

    def test_loads_all(self):
        ml.loads_all(ml.files.zincdb_fda_mol2.read_text(), "mol2", parser="molli")

    @ut.skipIf(not is_package_installed("openbabel"), "Openbabel is not installed")
    def test_loads_all_ob(self):
        ml.loads_all(ml.files.zincdb_fda_mol2.read_text(), "mol2", parser="molli")
        ml.loads_all(ml.files.zincdb_fda_molv3.read_text(), "mol", parser="openbabel")

    def test_dump(self):
        pass

    def test_dumps(self):
        pass

    def test_dump_all(self):
        pass

    def test_dumps_all(self):
        pass
