import unittest as ut
import numpy as np
import scipy.spatial.distance as spd
import molli_xt


class ExtensionTC(ut.TestCase):
    """This test suite is for the basic installation stuff"""

    def test_cdist3f(self):
        a1 = np.random.rand(5000, 3).astype(np.float32)
        a2 = np.random.rand(100, 3).astype(np.float32)

        d_molli = molli_xt.cdist22_eu2_f3(a1, a2)
        d_spd = spd.cdist(a1, a2, metric="sqeuclidean")

        self.assertAlmostEqual(np.max(d_molli - d_spd), 0, places=6)
