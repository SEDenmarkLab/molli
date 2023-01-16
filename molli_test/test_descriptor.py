import unittest as ut
import molli as ml


class DescriptorTC(ut.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_rectangular_grid(self):

        g1 = ml.descriptor.rectangular_grid([-1, 0, 0], [1, 0, 0], spacing=0.5)
        g2 = ml.descriptor.rectangular_grid([-1, -1, 0], [1, 1, 0], spacing=0.5)
        g3 = ml.descriptor.rectangular_grid(
            [-1, -1, -1], [1, 1, 1], spacing=0.5
        )

        self.assertTupleEqual(g1.shape, (5, 3))
        self.assertTupleEqual(g2.shape, (25, 3))
        self.assertTupleEqual(g3.shape, (125, 3))

        g4 = ml.descriptor.rectangular_grid(
            [-1, -1, -1], [1, 1, 1], spacing=0.5, padding=0.1
        )
        self.assertTupleEqual(g4.shape, (125, 3))
