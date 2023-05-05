from .. import MOLLI_USING_EXTENSIONS
import numpy as np
from numpy.typing import ArrayLike
from .rotation import rotation_matrix_from_axis


def _optimize_rotation(
    c1: ArrayLike, c2: ArrayLike, ax: ArrayLike, resolution: int = 12
):
    if not MOLLI_USING_EXTENSIONS:
        raise NotImplementedError
    else:
        import molli_xt as xt

    angles = np.radians(np.arange(0, 360, step=360 // resolution))
    aug_c2 = np.array([c2 @ rotation_matrix_from_axis(ax, ang) for ang in angles])

    dist = xt.cdist32_eu2_f3(aug_c2, c1)
    loss = np.sum(1 / dist, axis=(1, 2))

    best_angle = angles[np.argmin(loss)]
    return rotation_matrix_from_axis(ax, best_angle)
