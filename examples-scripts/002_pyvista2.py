import molli as ml
import pyvista as pv
from PIL import ImageColor
from matplotlib.colors import ListedColormap
import numpy as np

with ml.files.mol2.pdb_4a05.open() as f, ml.aux.timeit("Parsing mol2"):
    _mol = ml.chem.Molecule.load_mol2(f)

# mol = _mol.heavy
mol = _mol

for a in mol.atoms:
    if a.element.symbol == "N":
        a.atype = ml.AtomType.Dummy

mol.translate(-1 * mol.centroid())
a_sizes = [a.cov_radius_1 for a in mol.atoms]

# sph = pv.Sphere(theta_resolution=24, phi_resolution=24)

plotter = pv.Plotter(line_smoothing=True)

atoms = pv.MultiBlock()
for i, a in enumerate(mol.atoms):
    a_size = a.cov_radius_1 * 0.4
    if a.is_dummy:
        atoms.append(pv.Cube(mol.coords[i], a_size, a_size, a_size))
    else:
        atoms.append(
            pv.Sphere(
                a_size / 2,
                center=mol.coords[i],
                phi_resolution=32,
                theta_resolution=32,
            )
        )

bonds = pv.MultiBlock()
for j, b in enumerate(mol.bonds):
    r1, r2 = mol.bond_coords(b)
    bonds.append(pv.Tube(r1, r2, radius=0.03, n_sides=16))


actor, mapper = plotter.add_composite(
    atoms,
    smooth_shading=True,
    culling=True,
    ambient=0.2,
    diffuse=0.6,
    specular=0.2,
)

for i, a in enumerate(mol.atoms):
    mapper.block_attr[i + 1].color = a.color_cpk

actor, mapper = plotter.add_composite(
    bonds,
    color=(0.9, 0.9, 0.9),
    smooth_shading=True,
    culling=True,
    diffuse=0.9,
    specular=0.1,
)

# plotter.enable_anti_aliasing(aa_type="fxaa")
plotter.background_color = "000002"
plotter.view_xy()
plotter.add_axes()
plotter.show()
