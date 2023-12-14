"""
Create a grid to be subsequently used in grid based descriptor calculations.
This routine creates an hdf5 file that also store intermediate calculation results
such as grid
"""

from argparse import ArgumentParser
import molli as ml
from tqdm import tqdm
import zipfile
import msgpack
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import h5py
import fasteners

arg_parser = ArgumentParser(
    "molli grid",
    description="Read a molli library and calculate a grid",
)

arg_parser.add_argument(
    "library",
    action="store",
    help="Conformer library file to perform the calculations on",
)

arg_parser.add_argument(
    "-o",
    "--output",
    action="store",
    metavar="<fpath>",
    default=None,
    help="Destination for calculation results",
)

# arg_parser.add_argument(
#     "-f",
#     "--format",
#     action="store",
#     # metavar="fmt",
#     choices=("npy",),
#     default="npy",
#     help="Select the format that will be used for data storage.",
# )

arg_parser.add_argument(
    "-n",
    "--nprocs",
    action="store",
    default=1,
    type=int,
    help="Specifies the number of jobs for constructing a grid",
)

arg_parser.add_argument(
    "-p",
    "--padding",
    action="store",
    metavar="0.0",
    default=0.0,
    type=float,
    help="The bounding box will be padded by this many angstroms prior to grid construction",
)

arg_parser.add_argument(
    "-s",
    "--spacing",
    action="store",
    metavar="1.0",
    default=1.0,
    type=float,
    help="Intervals at which the grid points will be placed",
)

arg_parser.add_argument(
    "-b",
    "--batchsize",
    action="store",
    default=32,
    type=int,
    help="Number of molecules to be treated simulateneously",
)

arg_parser.add_argument(
    "--prune",
    action="store",
    nargs="?",
    const="2.0:0.5",
    default=False,
    metavar="<max_dist>:<eps>",
    help="Obtain the pruning indices for each conformer ensemble",
)

arg_parser.add_argument(
    "--indicator",
    action="store_true",
    help="Obtain the indicator indices for each conformer ensemble",
)


arg_parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite the existing grid file",
)

arg_parser.add_argument(
    "--dtype",
    type=np.dtype,
    default="int32",
    help="Specify the data format to be used for grid parameter storage.",
)


@delayed
def _min_max(_lib: ml.ConformerLibrary, keys: list[str]):
    with _lib.reading():
        ensembles = list(map(_lib.__getitem__, keys))

    mins = np.empty((len(keys), 3), dtype=np.float32)
    maxs = np.empty((len(keys), 3), dtype=np.float32)

    for i, ens in enumerate(ensembles):
        mins[i] = np.min(ens._coords, axis=(0, 1))
        maxs[i] = np.max(ens._coords, axis=(0, 1))

    return np.min(mins, axis=0), np.max(maxs, axis=0)


@delayed
def _pruning(
    _lib: ml.ConformerLibrary,
    grid: np.ndarray,
    keys: list[str],
    max_dist=2.0,
    eps=0.5,
):
    with _lib.reading():
        ensembles = list(map(_lib.__getitem__, keys))

    results = {
        k: ml.descriptor.prune(grid, ens, max_dist=max_dist, eps=eps)
        for k, ens in zip(keys, ensembles)
    }

    return results


@delayed
def _indicator(
    _lib: ml.ConformerLibrary,
    _gfpath: str | Path,
    keys: list[str],
    max_dist=2.0,
    dtype=np.int32,
):
    _lk_path = ml.aux.molli_aux_dir(_gfpath) / (_gfpath.name + ".lock")
    lock = fasteners.InterProcessReaderWriterLock(_lk_path)
    with lock.read_lock():
        with h5py.File(_gfpath, mode="r") as h5f:
            grid = np.asarray(h5f["grid"])
            if "grid_pruned" not in h5f.keys():
                raise RuntimeError("Cannot work with grids that haven't been pruned")

            if "indicator_pruned" in h5f.keys():
                tbd = [k for k in keys if k not in h5f["indicator_pruned"].keys()]
            else:
                tbd = keys

            pruned = {k: np.asarray(h5f["grid_pruned"][k]) for k in tbd}

    with _lib.reading():
        ensembles = list(map(_lib.__getitem__, tbd))

    results = {
        k: ml.descriptor.indicator(grid[pruned[k]], ens, max_dist=max_dist)
        for k, ens in zip(tbd, ensembles)
    }

    with lock.write_lock():
        with h5py.File(_gfpath, mode="a") as h5f:
            g = h5f.require_group("indicator_pruned")
            for k, ind in results.items():
                g.create_dataset(k, data=ind, dtype=dtype)


def molli_main(args, **kwargs):
    parsed = arg_parser.parse_args(args)

    if parsed.output is None:
        out_name = Path(parsed.library).stem + "_grid.hdf5"
        out_path = Path(parsed.library).parent / out_name
    else:
        out_path = Path(parsed.output)

    print(f"Using output file: {out_path.as_posix()}")

    parallel = Parallel(
        n_jobs=parsed.nprocs,
        backend="loky",
        return_as="generator",
    )

    library = ml.ConformerLibrary(parsed.library, readonly=True)

    with library.reading():
        keys = sorted(library.keys())

    if not parsed.overwrite and out_path.is_file():
        with h5py.File(out_path, "r") as f:
            if "grid" in f.keys():
                grid = np.asarray(f["grid"])
                qmin, qmax = f["grid"].attrs["bbox"]
                print(
                    "Successfully imported grid and bbox data from the previous calculation"
                )

    else:
        qmin, qmax = np.zeros(3), np.zeros(3)
        for cmin, cmax in tqdm(
            parallel(
                _min_max(library, batch)
                for batch in ml.aux.batched(keys, parsed.batchsize)
            ),
            desc="Computing a bounding box",
            total=ml.aux.len_batched(keys, parsed.batchsize),
        ):
            qmin = np.where(qmin < cmin, qmin, cmin)
            qmax = np.where(qmax > cmax, qmax, cmax)

        grid = ml.descriptor.rectangular_grid(
            qmin, qmax, parsed.padding, parsed.spacing
        )
        print(f"Finished calculating the bounding box!")

        with h5py.File(out_name, mode="w" if parsed.overwrite else "a") as f:
            f.create_dataset("grid", data=grid, dtype="f4")
            f["grid"].attrs["bbox"] = [qmin, qmax]

    V = np.prod(np.abs(qmax - qmin))
    q1 = np.array2string(qmin, precision=3, floatmode="fixed")
    q2 = np.array2string(qmax, precision=3, floatmode="fixed")
    print(
        f"Vectors: {q1}, {q2}. Number of grid points: {grid.shape[0]}. Volume: {V:0.2f} A**3."
    )

    if parsed.prune:
        max_dist, eps = map(float, parsed.prune.split(":"))
        print(f"Requested to calculate grid pruning with {max_dist=:0.3f} {eps=:0.3f}")

        with h5py.File(out_path, "a") as f:
            g = f.create_group("grid_pruned")
            for result in tqdm(
                parallel(
                    _pruning(library, grid, batch, max_dist=max_dist, eps=eps)
                    for batch in ml.aux.batched(keys, parsed.batchsize)
                ),
                desc="Computing pruned grids",
                total=ml.aux.len_batched(keys, parsed.batchsize),
            ):
                for key, pruned in result.items():
                    g.create_dataset(key, data=pruned, dtype=parsed.dtype)

    if parsed.indicator:
        for result in tqdm(
            parallel(
                _indicator(
                    library,
                    out_path,
                    batch,
                    max_dist=2.0,
                    dtype=parsed.dtype,
                )
                for batch in ml.aux.batched(keys, parsed.batchsize)
            ),
            desc="Computing indicator grids",
            total=ml.aux.len_batched(keys, parsed.batchsize),
        ):
            pass

    # np.save(parsed.output, grid, allow_pickle=False)

    # with ml.chem.ConformerLibrary(parsed.library, readonly=True) as lib:
    #     mins = np.zeros((len(lib), 3), dtype=np.float32)
    #     maxs = np.zeros((len(lib), 3), dtype=np.float32)

    #     for i, ens in enumerate(tqdm(lib, dynamic_ncols=True)):
    #         mins[i] = np.min(ens._coords, axis=(0, 1))
    #         maxs[i] = np.max(ens._coords, axis=(0, 1))

    # q1, q2 = np.min(mins, axis=0), np.max(maxs, axis=0)
    # grid = ml.descriptor.rectangular_grid(q1, q2, parsed.padding, parsed.spacing)
    # print(grid.shape)

    # np.save(parsed.output, grid, allow_pickle=False)
