"""
Compute descriptors from a library of conformers.
Note: the request for parallelized computation is accomodated with the best possible implementation,
but not guaranteed.
"""
from argparse import ArgumentParser

# from pprint import pprint
import os
import molli as ml
from tqdm import tqdm

from dask import distributed, delayed
import numpy as np
import h5py

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

#Potentially useful function depending on what needs to be written
# arg_parser.add_argument(
#     '-o',
#     '--output',
#     action='store',
#     type=str,
#     default=None,
#     metavar='<fpath>',  # ?
#     help='File path for directory to write to. Defaults to "aso.h5" in same directory as alt_aso.py'
# )

def molli_main(args, config=None, output=None, **kwargs):
    parsed = arg_parser.parse_args(args)

    print(
        f"Will compute descriptor {'w' if parsed.weighted else ''}{parsed.descriptor} using {parsed.nprocs} cores."
    )

    grid: np.ndarray = np.load(parsed.grid)
    print(f"Grid shape: {grid.shape}")

    # This could be the spot to implement instead of aa.aso_description, using ml.descriptor.aso2
    # Since ASO2 returns the np.average rather than the aa.aso_description that automatically writes the h5py file
    # if parsed.descriptor == 'ASO':
        # aa.aso_description(parsed.conflib, grid, parsed.output, parsed.nprocs, parsed.batchsize, parsed.chunksize)

    cluster = distributed.LocalCluster(n_workers=parsed.nprocs, processes=False)
    client = distributed.Client(cluster)

    lib = ml.chem.ConformerLibrary(parsed.conflib)
    nb = len(lib) // parsed.batchsize + 1

    print("Allocating storage for descriptors")

    strdt = h5py.special_dtype(vlen=str)

    with h5py.File("test_descriptor.h5", "w") as f:
    #Potential output use
    # with h5py.File(parsed.output, "w") as f:
        handles = f.create_dataset("handles", dtype=strdt, shape=(len(lib),))
        handles[:] = lib._block_keys

        data = f.create_dataset("descriptor", dtype="f4", shape=(len(lib), grid.shape[0]))
        data[:] = -1.0       

    for bn, batch in enumerate(tqdm(lib.yield_in_batches(parsed.batchsize), total=nb)):
        # with ml.aux.timeit(f"Processing batch {bn + 1} / {nb}"):

        scattered_chunk = client.scatter(batch)


        futures = client.map(
            lambda x: ml.descriptor.chunky_aso(x, grid, chunksize=parsed.chunksize),
            scattered_chunk,
        )


        # distributed.progress(futures)
        asos = client.gather(futures)


        _start = bn * parsed.batchsize
        _end = _start + len(batch)


        with h5py.File("test_descriptor.h5", "r+") as f:
        #Potential Output use
        # with h5py.File(parsed.output, "r+") as f:
            f["descriptor"][_start:_end] = asos
            
    
        