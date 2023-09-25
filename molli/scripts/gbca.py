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
from mpi4py import MPI
import numpy as np
import h5py
from logging import getLogger
from uuid import uuid1
import fasteners
import tempfile
import shutil

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
    help="Apply the weights specified in the conformer files",
)

arg_parser.add_argument(
    "-n",
    "--nprocs",
    action="store",
    type=int,
    metavar=OS_NCORES,
    default=OS_NCORES,
    help="Selects number of processors for python multiprocessing application. "
         "If the program is launched via MPI backend, this parameter is ignored.",
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
    help="Number of conformer ensembles to be processed in one batch.",
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

def _mpi_worker():
    ...



def molli_main(args, shared_lkfile=None, scratch_lkfile=None, **kwargs):
    parsed = arg_parser.parse_args(args)
    logger = getLogger("molli.scripts.gbca")
    
    # Test if molli is launched via openmpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size > 1:
        # This should only happen if MPI launch detected.
        if rank == 0:
            print(ml.config.SPLASH)
            logger.info(f"MPI launch detected: universe size {size}. --nprocs CMD parameter ignored even if specified.")
            logger.debug(f"Shared  lock file: {shared_lkfile}")
            logger.debug(f"Scratch lock file: {scratch_lkfile}")
        
        lib = ml.ConformerLibrary(parsed.conflib)
        grid = np.load(parsed.grid)

        # This enables strided access to the batch
        for batch_idx in range(rank, lib.n_batches(parsed.batchsize), size):
            try:
                with ml.ConformerLibrary.new(ml.config.SCRATCH_DIR / f"molli-gbca-{batch_idx}-{rank}.mlib", overwrite=True) as templib, lib:
                    pos = batch_idx * parsed.batchsize
                    chunksize = min(len(lib) - pos, parsed.batchsize)
                    # This creates a
                    lib.copy_chunk(templib, pos, chunksize)
            except Exception as xc:
                logger.critical("Could not copy batch {batch_idx}")
                logger.exception(xc)
                comm.Abort(1)
            logger.info(f"Batch {batch_idx} was copied into {templib.path}")
            with h5py.File(ml.config.SCRATCH_DIR / f"aso-{batch_idx}-{rank}.hdf5", "w") as f, templib:
                for key in templib.keys():
                    f[key] = ml.descriptor.chunky_aso(templib[key], grid, parsed.chunksize)

            logger.info(f"Computing ASO for batch {batch_idx} (last key: {key}) finished.")
        comm.barrier()
        if rank == 0:
            logger.info(f"Scattering the library across all workers is finished.")
        
    else:
        # Using python multiprocessing now.
        logger.info(f"Using python multiprocessing with {parsed.nprocs} processes.")
        
    # print(
    #     f"Will compute descriptor {'w' if parsed.weighted else ''}{parsed.descriptor} using {parsed.nprocs} cores."
    # )
    #         
     