# This file provides the code used in the benchmarking of molli cdist vs scipy cdist implementations.
import molli as ml
import h5py
from tqdm import tqdm
import numpy as np

with h5py.File("box_aligned_grid.hdf5") as f:
    grid32 = np.asarray(f["grid"], dtype="float32")
    grid64 = np.asarray(f["grid"], dtype="float64")

lib = ml.ConformerLibrary("box_aligned_1.mlib")
with lib.reading(), open("result_descript.csv", "wt") as outf:
    for ens in tqdm(lib.values(), total=len(lib)):
        with ml.aux.timeit(print_on_exit=False) as t1:
            ref_aso = ml.descriptor.aso(
                ens,
                grid64,
                dtype="float64",
            )

        with ml.aux.timeit(print_on_exit=False) as t2:
            faster_aso = ml.descriptor.aso(
                ens,
                grid32,
                dtype="float32",
            )

        diff = np.abs(ref_aso - faster_aso)

        maxd = np.max(diff)
        norm = np.max(diff) / np.max(ref_aso)

        outf.write(
            f"{ens.name},{ens.coords.size},{t1.td_ns},{t2.td_ns},{maxd:.2e},{norm:.2e}\n"
        )
