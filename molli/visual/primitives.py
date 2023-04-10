import pyvista as pv
import molli as ml
import numpy as np
from PIL import ImageColor


def color_cpk(a: ml.Atom):

    if a.is_dummy:
        return [1, 0, 0]
    elif a.is_attachment_point:
        return [1, 1, 0]
    else:
        r, g, b = ImageColor.getrgb(a.color_cpk)
        return [r / 255, g / 255, b / 255]


def colors_cpk(s: ml.StructureLike):
    colors = np.empty((s.n_atoms, 3))
    for i, a in enumerate(s.atoms):
        colors[i] = color_cpk(a)
    return colors


def wireframe(struct: ml.StructureLike) -> pv.PolyData:
    """
    This creates a wireframe out of a structure-like object
    """
    lines = []
    for b in struct.bonds:
        i1, i2 = struct.get_atom_indices(b.a1, b.a2)
        lines.extend((2, i1, i2))

    wire = pv.PolyData(struct.coords, n_lines=len(lines), lines=lines)
    wire.point_data["color_cpk"] = colors_cpk(struct)
    wire.point_data["radius_cov"] = [a.cov_radius_1 or 0.5 for a in struct.atoms]
    return wire


def atoms(struct: ml.StructureLike) -> list[pv.PolyData]:
    mb_atoms = pv.MultiBlock()
    for i, a in enumerate(struct.atoms):
        if cov_rad := a.cov_radius_1:
            a_size = cov_rad * 0.4
        else:
            a_size = 0.2
        if a.is_dummy:
            mesh = pv.Cube(struct.coords[i], a_size, a_size, a_size)
        if a.is_attachment_point:
            a2 = next(struct.connected_atoms(a))
            v = struct.vector(a2, a)
            o = struct.get_atom_coord(a2)
            mesh = pv.Arrow(
                o + v / 2, v / 2, shaft_radius=a_size / 2, tip_radius=a_size
            )
        else:
            mesh = pv.Sphere(
                a_size,
                center=struct.coords[i],
                phi_resolution=32,
                theta_resolution=32,
            )
        mesh.point_data["color_cpk"] = [color_cpk(a)] * mesh.n_points
        mb_atoms.append(mesh)

    return mb_atoms


def test_ballnstick(s: ml.StructureLike):
    plt = pv.Plotter()
    w = wireframe(s)
    # w.point_data["color_cpk"] = colors_cpk(s)
    # w.point_data["radius_cov"] = [a.cov_radius_1 or 0.5 for a in s.atoms]
    # w.point_data["scalars"] = [glyph_type(a) for a in s.atoms]

    tubes = w.tube(radius=0.05)
    spheres = w.glyph(orient=False, scale="radius_cov", geom=pv.Sphere(radius=0.4))

    # plt.add_mesh(
    #     tubes,
    #     rgb=True,
    #     scalars="color_cpk",
    #     smooth_shading=True,
    #     interpolate_before_map=False,
    #     culling=True,
    # )

    plt.add_mesh(
        tubes + spheres,
        rgb=True,
        scalars="color_cpk",
        smooth_shading=True,
        interpolate_before_map=False,
    )

    plt.add_point_labels(
        w, [a.label or a.element.symbol for a in s.atoms], always_visible=True
    )
    plt.enable_ssao()
    plt.show()
