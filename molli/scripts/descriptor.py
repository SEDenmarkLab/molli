"""
Compute descriptors from a collection
"""
from argparse import ArgumentParser

# from pprint import pprint
import os
import molli as ml
from tqdm import tqdm

from dask import distributed, delayed
import numpy as np

DESCRIPTOR_CHOICES = ["ASO", "AEIF", "ADIF", "AESP", "ASR"]

OS_NCORES = os.cpu_count() // 2

arg_parser = ArgumentParser(
    "molli descriptor",
    description="This module can be used for standalone computation of descriptors",
)

arg_parser.add_argument(
    "descriptor",
    choices=DESCRIPTOR_CHOICES,
    type=str.upper,
    help="This selects the specific descriptor to compute.",
)
arg_parser.add_argument(
    "conflib",
    metavar="MLIB_FILE",
    action="store",
    type=str,
    help="Conformer library to perform the calculation on",
)

arg_parser.add_argument(
    "-w",
    "--weighted",
    action="store_true",
    help="Print molli configuration",
)


arg_parser.add_argument(
    "-n",
    "--nprocs",
    action="store",
    type=int,
    metavar=OS_NCORES,
    default=OS_NCORES,
    help="Selects number of processors",
)

arg_parser.add_argument(
    "-c",
    "--chunksize",
    action="store",
    type=int,
    metavar=512,
    default=512,
    help="Selects number of processors",
)


arg_parser.add_argument(
    "-b",
    "--batchsize",
    action="store",
    type=int,
    metavar=128,
    default=128,
    help="Number of conformer ensembles in one batch",
)


arg_parser.add_argument(
    "-g",
    "--grid",
    action="store",
    metavar="<grid.npy>",
    type=str,
    default=None,
    help="Selects the locations of grid points.",
)


def molli_main(args, config=None, output=None, **kwargs):
    parsed = arg_parser.parse_args(args)

    print(
        f"Will compute descriptor {'w' if parsed.weighted else ''}{parsed.descriptor} using {parsed.nprocs} cores."
    )

    grid: np.ndarray = np.load(parsed.grid)
    print(f"Grid shape: {grid.shape}")

    cluster = distributed.LocalCluster(n_workers=parsed.nprocs)
    client = distributed.Client(cluster)

    lib = ml.chem.ConformerLibrary(parsed.conflib)
    nb = len(lib) // parsed.batchsize + 1
    for bn, batch in enumerate(lib.yield_in_batches(parsed.batchsize)):
        with ml.aux.timeit(f"Processing batch {bn + 1} / {nb}"):
            futures = client.map(
                lambda x: ml.descriptor.chunky_aso(x, grid, chunksize=parsed.chunksize),
                batch,
            )
            distributed.progress(futures)

    # with ml.chem.ConformerLibrary(parsed.conflib) as lib:
    #     for ens in tqdm(lib):
    #         # ml.descriptor.parallel_aso(
    #         #     ens,
    #         #     grid,
    #         #     n_threads=parsed.nprocs,
    #         #     chunksize=parsed.chunksize,
    #         # )
    #         ml.descriptor.chunky_aso(
    #             ens,
    #             grid,
    #             chunksize=parsed.chunksize,
    #         )
