"""
Grid based conformer averaged descriptors
This is a foundational file
"""
import numpy as np
from numpy.typing import ArrayLike


def rectangular_grid(
    q1: ArrayLike,
    q2: ArrayLike,
    padding: float = 0.0,
    spacing: float = 1.0,
    dtype: str = "float32",
) -> np.ndarray:
    l = np.array(q1, dtype=dtype) - padding
    r = np.array(q2, dtype=dtype) + padding

    # Number of points
    nx = int((r[0] - l[0]) // spacing)
    ny = int((r[1] - l[1]) // spacing)
    nz = int((r[2] - l[2]) // spacing)

    # Offsets
    ox = (r[0] - l[0] - nx * spacing) / 2
    oy = (r[1] - l[1] - nx * spacing) / 2
    oz = (r[2] - l[2] - nx * spacing) / 2

    # X linsp
    xs = np.linspace(l[0] + ox, r[0] - ox, nx, dtype=dtype)
    ys = np.linspace(l[1] + ox, r[1] - ox, nx, dtype=dtype)
    zs = np.linspace(l[2] + ox, r[2] - ox, nx, dtype=dtype)

    xx, yy, zz = np.meshgrid(xs, ys, zs)

    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


# class Grid:
#     def __init__(self, grid: ArrayLike = None, dtype: str = "float32"):
#         self._dtype = dtype
#         self.grid = np.array(grid, dtype=self._dtype)

#     @property
#     def grid(self):
#         return self._grid

#     @grid.setter
#     def grid(self, other: ArrayLike | None):
#         self._grid = np.array(other, self._dtype)

#     @property
#     def n_gridpoints(self):
#         return self._grid.shape[0]
