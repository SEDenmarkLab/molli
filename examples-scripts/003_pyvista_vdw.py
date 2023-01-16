import molli as ml
import pyvista as pv
from PIL import ImageColor
from matplotlib.colors import ListedColormap
import numpy as np

pv.set_plot_theme("dark")  # "document" for light settings!
# print(pv.Report())

with ml.files.mol2.dendrobine.open() as f:
    _mol = ml.chem.Molecule.load_mol2(f)

mol = _mol.heavy
# mol = _mol

mol.translate(-1 * mol.centroid())
a_sizes = [a.cov_radius_1 for a in mol.atoms]

sph = pv.Sphere(theta_resolution=32, phi_resolution=32)


plotter = pv.Plotter(notebook=False)


b_lines = []
for b in mol.bonds:
    i1, i2 = map(mol.get_atom_index, (b.a1, b.a2))
    b_lines.append((2, i1, i2))

points = pv.PolyData(mol.coords, lines=b_lines, n_lines=mol.n_bonds)
points.point_data["element"] = [a.Z for a in mol.atoms]
points["radius"] = a_sizes
points["vdw_radius"] = [a.vdw_radius for a in mol.atoms]


val = np.linspace(-1, 118, 120)
colors = np.zeros((120, 4))
for i, elt in enumerate(ml.chem.Element):
    clr = elt.color_cpk or "#000000"
    r, g, b = ImageColor.getrgb(clr)
    colors[val > (elt.z - 0.5)] = [r / 255, g / 255, b / 255, 1]

cmap_cpk = pv.LookupTable(
    ListedColormap(colors), n_values=120, scalar_range=(-1, 118)
)

spherez = points.glyph(
    geom=sph,
    orient=False,
    scale="radius",
    factor=0.75,
    progress_bar=True,
)

tubez = points.tube(
    radius=0.05,
    progress_bar=True,
    n_sides=24,
    capping=False,
)


plotter.add_mesh(
    spherez,
    color="white",
    smooth_shading=True,
    scalars="element",
    show_scalar_bar=False,
    diffuse=0.7,
    ambient=0.1,
    specular=0.2,
    specular_power=5,
    cmap=cmap_cpk,
    culling=True,
    # opacity=0.5,
)

# tritubes = tubez.triangulate()
# tritubes_trunc = tritubes.boolean_difference(spherez)

heavy = pv.PolyData(mol.heavy.coords)
heavy["labels"] = [
    f"{a.element.symbol}({i})" for i, a in enumerate(mol.heavy.atoms)
]

plotter.add_mesh(
    tubez,
    color="white",
    smooth_shading=True,
    scalars="element",
    show_scalar_bar=False,
    diffuse=0.7,
    ambient=0.1,
    specular=0.2,
    specular_power=5,
    interpolate_before_map=False,
    cmap=cmap_cpk,
    culling=True,
    # opacity=0.5,
    pickable=True,
)


plotter.add_point_labels(
    heavy,
    "labels",
    font_size=20,
    shadow=True,
    shape_color="white",
    shape_opacity=0.25,
    show_points=False,
    always_visible=True,
    margin=5,
    # font_family="courier",
)

import molli as ml
import pyvista as pv
from PIL import ImageColor
from matplotlib.colors import ListedColormap
import numpy as np

pv.set_plot_theme("dark")  # "document" for light settings!
# print(pv.Report())

with ml.files.mol2.dendrobine.open() as f:
    _mol = ml.chem.Molecule.load_mol2(f)

mol = _mol.heavy
# mol = _mol

mol.translate(-1 * mol.centroid())
a_sizes = [a.cov_radius_1 for a in mol.atoms]

sph = pv.Sphere(theta_resolution=32, phi_resolution=32)


plotter = pv.Plotter(notebook=False)


b_lines = []
for b in mol.bonds:
    i1, i2 = map(mol.get_atom_index, (b.a1, b.a2))
    b_lines.append((2, i1, i2))

points = pv.PolyData(mol.coords, lines=b_lines, n_lines=mol.n_bonds)
points.point_data["element"] = [a.Z for a in mol.atoms]
points["radius"] = a_sizes
points["vdw_radius"] = [a.vdw_radius for a in mol.atoms]


val = np.linspace(-1, 118, 120)
colors = np.zeros((120, 4))
for i, elt in enumerate(ml.chem.Element):
    clr = elt.color_cpk or "#000000"
    r, g, b = ImageColor.getrgb(clr)
    colors[val > (elt.z - 0.5)] = [r / 255, g / 255, b / 255, 1]

cmap_cpk = pv.LookupTable(
    ListedColormap(colors), n_values=120, scalar_range=(-1, 118)
)

spherez = points.glyph(
    geom=sph,
    orient=False,
    scale="radius",
    factor=0.75,
    progress_bar=True,
)

tubez = points.tube(
    radius=0.05,
    progress_bar=True,
    n_sides=24,
    capping=False,
)


plotter.add_mesh(
    spherez,
    color="white",
    smooth_shading=True,
    scalars="element",
    show_scalar_bar=False,
    diffuse=0.7,
    ambient=0.1,
    specular=0.2,
    specular_power=5,
    cmap=cmap_cpk,
    culling=True,
    # opacity=0.5,
)

# tritubes = tubez.triangulate()
# tritubes_trunc = tritubes.boolean_difference(spherez)

heavy = pv.PolyData(mol.heavy.coords)
heavy["labels"] = [
    f"{a.element.symbol}({i})" for i, a in enumerate(mol.heavy.atoms)
]

plotter.add_mesh(
    tubez,
    color="white",
    smooth_shading=True,
    scalars="element",
    show_scalar_bar=False,
    diffuse=0.7,
    ambient=0.1,
    specular=0.2,
    specular_power=5,
    interpolate_before_map=False,
    cmap=cmap_cpk,
    culling=True,
    # opacity=0.5,
    pickable=True,
)


plotter.add_point_labels(
    heavy,
    "labels",
    font_size=20,
    shadow=True,
    shape_color="white",
    shape_opacity=0.25,
    show_points=False,
    always_visible=True,
    margin=5,
    # font_family="courier",
)

vdw_surf = pv.Sphere(center=mol.coords[0], radius=mol.atoms[0].vdw_radius)

for i in range(1, 2):
    new_sphere = pv.Sphere(center=mol.coords[i], radius=mol.atoms[i].vdw_radius)
    outer = new_sphere - vdw_surf
    vdw_surf += outer
    vdw_surf.clean()
    vdw_surf.triangulate()


plotter.add_mesh(
    vdw_surf,
    color="white",
    style="wireframe",
    smooth_shading=True,
    show_scalar_bar=False,
    diffuse=0.7,
    ambient=0.1,
    specular=0.2,
    specular_power=5,
    cmap=cmap_cpk,
    culling=True,
    opacity=0.2,
)


plotter.enable_anti_aliasing(aa_type="fxaa", multi_samples=4)
plotter.view_xy()
plotter.add_axes()
plotter.enable_ssao(kernel_size=32)
plotter.enable_depth_peeling()
# plotter.enable_point_picking(left_clicking=True)
# plotter.enable_stereo_render()
plotter.show()
