import molli as ml
import pyvista as pv
from PIL import ImageColor
from matplotlib.colors import ListedColormap
import numpy as np

with ml.files.mol2.dendrobine.open() as f:
    _mol = ml.chem.Molecule.from_mol2(f)

# mol = _mol.heavy
mol = _mol

mol.translate(-1 * mol.centroid())
a_sizes = [a.cov_radius_1 for a in mol.atoms]

sph = pv.Sphere(theta_resolution=24, phi_resolution=24)


plotter = pv.Plotter()

b_lines = []
for b in mol.bonds:
    i1, i2 = map(mol.get_atom_index, (b.a1, b.a2))
    b_lines.append((2, i1, i2))

points = pv.PolyData(mol.coords, lines=b_lines, n_lines=mol.n_bonds)
points.point_data["element"] = [a.Z for a in mol.atoms]
points["radius"] = a_sizes

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
    # progress_bar=True,
)

tubez = points.tube(
    radius=0.05,
    # progress_bar=True,
)

plotter.add_mesh(
    spherez,
    color="white",
    smooth_shading=True,
    scalars="element",
    show_scalar_bar=False,
    diffuse=0.7,
    specular=0.3,
    specular_power=2,
    cmap=cmap_cpk,
)

plotter.add_mesh(
    tubez,
    color="white",
    smooth_shading=True,
    scalars="element",
    show_scalar_bar=False,
    diffuse=0.7,
    specular=0.3,
    specular_power=2,
    interpolate_before_map=False,
    cmap=cmap_cpk,
)


plotter.enable_anti_aliasing(aa_type="ssaa", multi_samples=4)
plotter.background_color = "000002"
plotter.view_xy()
plotter.add_axes()
plotter.show()
