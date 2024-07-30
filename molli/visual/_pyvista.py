import molli as ml
import pyvista as pv
from PIL import ImageColor
import numpy as np

BGCOLOR = "black"
THEME = "dark"
WIDTH = 800
HEIGHT = 400


def draw_ballnstick(
    s: ml.Structure,
    plt=None,
    quality: int = 16,
    opacity: float = None,
):
    _tobeshown = False
    if plt is None:
        _tobeshown = True
        plt = pv.Plotter()
        plt.set_background(BGCOLOR)

    # This is the basic shape for an atom
    # Later: add the same for attachment points and such
    atom_glyph = pv.Sphere(radius=0.3, theta_resolution=quality, phi_resolution=quality)
    dummy_glyph = pv.Tetrahedron(radius=0.3)

    # Bin the atoms based on their element identity
    atom_elt_bins: dict[ml.Element, list[int]] = {}
    for i, a in enumerate(s.atoms):
        e = a.element
        if e in atom_elt_bins:
            atom_elt_bins[e].append(a.idx)
        else:
            atom_elt_bins[e] = [a.idx]

    for element, atom_id_list in atom_elt_bins.items():
        substr = ml.Substructure(s, atom_id_list)
        mesh = pv.PolyData(substr.coords).glyph(
            factor=element.cov_radius_1 or 1,
            geom=atom_glyph if element > 0 else dummy_glyph,
        )
        plt.add_mesh(
            mesh,
            color=element.color_cpk if element > 0 else "navy",
            smooth_shading=True,
            diffuse=0.60,
            ambient=0.40,
            opacity=opacity,
            # silhouette=True,
        )

    lines = [(2, b.a1.idx, b.a2.idx) for b in s.bonds]

    if lines:

        tubes = pv.PolyData(s.coords, lines=lines).tube(radius=0.05, n_sides=12)

        plt.add_mesh(
            tubes,
            smooth_shading=True,
            color="silver",
            diffuse=0.60,
            ambient=0.40,
            opacity=opacity,
            # silhouette=True,
        )

    if _tobeshown:
        plt.show()


def draw_wireframe(
    s: ml.Structure | ml.ConformerEnsemble,
    plt=None,
    line_width: int = 2,
    opacity: float = 1.0,
    color_darkness: int = 0.0,
):
    _tobeshown = False
    if plt is None:
        _tobeshown = True
        plt = pv.Plotter()
        plt.set_background(BGCOLOR)

    if isinstance(s, ml.ConformerEnsemble):
        lines = []
        for i, conf in enumerate(s):
            for b in s.bonds:
                i1, i2 = map(s.get_atom_index, (b.a1, b.a2))
                lines.extend((2, i1 + s.n_atoms * i, i2 + s.n_atoms * i))

        colors = [ImageColor.getrgb(a.color_cpk) for a in s.atoms] * s.n_conformers
        colors = np.clip((np.array(colors) - color_darkness) / 255, 0, 1)

        data = pv.PolyData(
            s.coords.reshape((s.n_atoms * s.n_conformers, 3)),
            n_lines=s.n_bonds,
            lines=lines,
        )

    else:
        lines = []
        for b in s.bonds:
            i1, i2 = map(s.get_atom_index, (b.a1, b.a2))
            lines.extend((2, i1, i2))

        colors = np.empty((s.n_atoms, 3))

        colors = [ImageColor.getrgb(a.color_cpk) for a in s.atoms]
        colors = np.array(colors) / 255

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

    if _tobeshown:
        plt.show()


def plot_descriptor(
    grid: np.ndarray,
    values: np.ndarray,
    plt: pv.Plotter,
    name: str = "descriptor",
    style="spheres",
    radius: float = 0.5,
    factor: float = 1.0,
    opacity: float = 1.0,
    box: bool = True,
    cmap=None,
    scalar_bar_args: dict = None,
):
    match style:
        case "spheres":
            sph = pv.Sphere(radius)
            pd = pv.PolyData(grid)
            pd.point_data[name] = values
            gly = pd.glyph(geom=sph, scale=name, factor=factor)
            plt.add_mesh(
                gly, scalars=name, opacity=opacity, scalar_bar_args=scalar_bar_args
            )

            if box:
                x1, y1, z1 = np.min(grid, axis=0)
                x2, y2, z2 = np.max(grid, axis=0)

                bbox = pv.Cube(bounds=(x1, x2, y1, y2, z1, z2))
                plt.add_mesh(bbox, style="wireframe", line_width=1)

        case _:
            raise NotImplementedError
