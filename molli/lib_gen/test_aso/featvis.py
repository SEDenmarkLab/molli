import pyvista as pv
import numpy as np
import molli as ml
import molli.visual.backend_pyvista as mlvis
from tqdm import tqdm
# import yaml
from pathlib import Path
import h5py


grid_file = "grid.npy"
conformer_lib_file = "../../../out_conformers1/conformers-cmd-test.mlib"
aso_file = "massive_aso.h5"


# catalysts_used_in_modelling = {
#   '6_1_1_1', '1_1_1_1', 'aa_1', '5_1_1_2', '6_1_1_21', '1_1_1_3',
#   '190_1_1_2', '3_1_1_21', '1_1_1_22', '1_1_1_26', '1_1_1_24', '1_1_1_7',
#   '11_1_1_30', '1_1_1_30', '1_1_1_27', '6_1_1_14', '1_1_1_29', '1_1_1_15',
#   '1_1_1_20', '1_1_1_18', 'aa_18', '171_2_2_17', '90_1_1_17', 
#   '7_1_2_2', '14_1_1_13', '281_4_4_2', '22_4_4_28', '249_4_4_3', 
#   '227_2_2_2', '254_2_2_11', '16_3_1_9', '200_3_1_21',
#   '73_3_1_29', '14_1_2_14', '56_2_1_1', '7_2_1_2', '1_4_1_2',
#   '187_1_4_30','187_1_4_2', '185_2_1_10', '185_1_2_2', '154_1_2_15', '3_1_2_18',
#   '250_1_3_12', '252_1_1_8', '225_1_1_13', '225_1_1_2'
# }

root = Path(".")
grid_full = np.load(grid_file)

plt = pv.Plotter()

x1, y1, z1 = grid_full[0]
x2, y2, z2 = grid_full[-1]

lib = ml.ConformerLibrary(conformer_lib_file)

# print(lib)
# print([i.name for i in lib ])
# exit()

bbox = pv.Cube(bounds=(x1, x2, y1, y2, z1, z2))

geom_gridpoint = pv.Cube(x_length=0.1, y_length=0.1, z_length=0.1)
geom_feature = pv.Sphere(radius=0.5)

aso_accum = np.zeros(grid_full.shape[0], dtype="float32")

with h5py.File(aso_file) as f:
    for x in tqdm(f.keys(), desc="Accumulating ASO"):
        arr =np.array(f[x])
        aso_accum += arr

# grid_full = np.load("insilico_grid.npy")
grid_full_mesh = pv.PolyData(grid_full)  # .glyph(geom=geom_gridpoint)
grid_full_mesh.point_data["cumulative_aso"] = aso_accum
grid_full_gly = grid_full_mesh.glyph(geom=geom_feature, scale="cumulative_aso", factor=1e-5)


# for ens in tqdm([i for i in lib if i in catalysts_used_in_modelling]):  ###### SHAME! ######
for ens in tqdm(lib, desc="Adding structure wireframe"):
    # if ens.name in catalysts_used_in_modelling:
    mlvis.plot_structure_as_wireframe(plt, ens[0].heavy, 0.20)


plt.add_mesh(grid_full_mesh, style="points", color="gray", point_size=3, opacity=0.4)
plt.add_mesh(bbox, style="wireframe", line_width=3, color="green", opacity=0.5)
plt.add_mesh(grid_full_gly, scalars="cumulative_aso", opacity=0.5, smooth_shading=True)


plt.enable_anti_aliasing("fxaa")
plt.show_axes()
plt.set_background("black")
plt.show()
