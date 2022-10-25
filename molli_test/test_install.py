import unittest as ut


class BasicInstallTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_import_obabel_python_bindings(self):
        """Tests if python openbabel bindings can be imported"""
        import openbabel

    def test_import_molli(self):
        """Tests"""
        import molli

    def test_import_molli_extensions(self):
        """Tests"""
        import molli_xt
