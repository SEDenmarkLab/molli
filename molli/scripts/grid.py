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
    default="grid",
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
    "--n_jobs",
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
    action="store_true",
    help="Obtain the pruning indices for each conformer ensemble",
)

arg_parser.add_argument(
    "--indicator",
    action="store_true",
    help="Obtain the indicator indices for each conformer ensemble",
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


def molli_main(args, **kwargs):
    parsed = arg_parser.parse_args(args)

    parallel = Parallel(n_jobs=parsed.n_jobs, backend="loky")
    library = ml.ConformerLibrary(parsed.library, readonly=True)

    with library.reading():
        keys = sorted(library.keys())
        
    batches = list(ml._aux.batched(keys, parsed.batchsize))

    qmin, qmax = np.zeros(3), np.zeros(3)

    for cur_qmin, cur_qmax in tqdm(
        parallel(_min_max(library, batch) for batch in batches),
        total=len(batches),
        desc="Calculating the bounding box",
    ):
        qmin = np.where(cur_qmin < qmin, cur_qmin, qmin)
        qmax = np.where(cur_qmax > qmax, cur_qmax, qmax)

    grid = ml.descriptor.rectangular_grid(qmin, qmax, parsed.padding, parsed.spacing)
    print(f"Finished calculating the bounding box!")
    print(f"Vectors: {qmin=}, {qmax=}. Number of grid points: {grid.shape[0]}")

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
