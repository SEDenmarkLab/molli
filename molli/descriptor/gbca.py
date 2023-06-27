"""
Grid based conformer averaged descriptors
This is a foundational file
"""
import numpy as np
from numpy.typing import ArrayLike


def rectangular_grid(
    r1: ArrayLike,
    r2: ArrayLike,
    padding: float = 0.0,
    spacing: float = 1.0,
    dtype: str = "float32",
) -> np.ndarray:
    """
    Creates a rectangular grid of points

    Args:
        r1 (ArrayLike): First corner of the rectangle
        r2 (ArrayLike): Second corner of the rectangle
        padding (float, optional): Padding to add to the rectangle. Defaults to 0.0.
        spacing (float, optional): Spacing between gridpoints. Defaults to 1.0.
        dtype (str, optional): Data type of the gridpoints. Defaults to "float32".
    
    Returns:
        np.ndarray: Gridpoints
    """
    l = np.array(r1, dtype=dtype) - padding
    r = np.array(r2, dtype=dtype) + padding

    # Number of points
    nx = int((r[0] - l[0]) // spacing) + 1
    ny = int((r[1] - l[1]) // spacing) + 1
    nz = int((r[2] - l[2]) // spacing) + 1

    # Offsets
    ox = (r[0] - l[0] - (nx - 1) * spacing) / 2
    oy = (r[1] - l[1] - (ny - 1) * spacing) / 2
    oz = (r[2] - l[2] - (nz - 1) * spacing) / 2

    # X linsp
    xs = np.linspace(l[0] + ox, r[0] - ox, nx, endpoint=True, dtype=dtype)
    ys = np.linspace(l[1] + oy, r[1] - oy, ny, endpoint=True, dtype=dtype)
    zs = np.linspace(l[2] + oz, r[2] - oz, nz, endpoint=True, dtype=dtype)

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
