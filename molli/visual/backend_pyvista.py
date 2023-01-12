from ..chem import Structure
import pyvista as pv
from PIL import ImageColor
import numpy as np


def plot_structure_as_wireframe(plt: pv.Plotter, s: Structure, opacity=1.0):
    """This just shows a given structure as wireframe"""
    lines = []
    for b in s.bonds:
        i1, i2 = map(s.get_atom_index, (b.a1, b.a2))
        lines.extend((2, i1, i2))

    colors = np.empty((s.n_atoms, 3))

    for i, a in enumerate(s.atoms):
        colors[i] = ImageColor.getrgb(a.color_cpk)

    colors /= 255

    data = pv.PolyData(s.coords, n_lines=s.n_bonds, lines=lines)
    mesh_actor: pv.Actor = plt.add_mesh(
        data,
        rgb=True,
        scalars=colors,
        interpolate_before_map=False,
        line_width=2,
        opacity=opacity,
    )

    mesh_actor: pv.Actor = plt.add_points(
        data,
        rgb=True,
        scalars=colors,
        interpolate_before_map=False,
        point_size=3,
        opacity=opacity,
    )
