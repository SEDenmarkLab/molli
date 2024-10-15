import molli as ml
import timeit
import h5py
import numpy as np

_clib = ml.ConformerLibrary("bpa_test.clib")

with _clib.reading():
    ensembles = {k: v for k, v in _clib.items()}

path = "bpa_test.hdf5"


def write_hdf5():
    with h5py.File(path, "w") as f:
        for entry_key, ensemble in ensembles.items():
            group = f.create_group(entry_key)
            group.create_dataset(
                name="coords",
                data=ensemble.coords,
                dtype="float32",
            )

            group.create_dataset(
                name="atomic_charges", data=ensemble.atomic_charges, dtype="float32"
            )
            group.create_dataset(name="weights", data=ensemble.weights, dtype="float32")
            group.create_dataset(
                name="atoms",
                data=[int(a.element) for a in ensemble.atoms],
                dtype="int16",
            )
            atom_idxs = {a: i for i, a in enumerate(ensemble.atoms)}
            group.create_dataset(
                name="bonds",
                data=[
                    [atom_idxs[bond.a1], atom_idxs[bond.a2], int(bond.btype)]
                    for bond in ensemble.bonds
                ],
                dtype="int16",
            )


def read_hdf5():
    with h5py.File(path, "r") as f:
        for entry_key in f.keys():
            group = f[entry_key]
            ens = ml.ConformerEnsemble(
                [int(i) for i in group["atoms"][:]],
                n_conformers=group["coords"].shape[0],
                name=entry_key,
                coords=group["coords"][:],
                weights=group["weights"][:],
                atomic_charges=group["atomic_charges"][:],
            )
            for a1, a2, bt in group["bonds"][:]:
                ens.connect(int(a1), int(a2), btype=ml.BondType(bt))


# Measure writing speed
write_times = timeit.repeat(write_hdf5, repeat=5, number=1)
print(f"Writing time: min {min(write_times):.6f} seconds of {write_times}")

# Measure reading speed
read_times = timeit.repeat(read_hdf5, repeat=5, number=1)
print(f"Reading time: min {min(read_times):.6f} seconds of {read_times}")
