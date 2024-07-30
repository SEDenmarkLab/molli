# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Alexander S. Shved <shvedalx@illinois.edu>
#
# S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign
# https://denmarkgroup.illinois.edu/
#
# Copyright 2022-2023 The Board of Trustees of the University of Illinois.
# All Rights Reserved.
#
# Licensed under the terms MIT License
# The License is included in the distribution as LICENSE file.
# You may not use this file except in compliance with the License.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# ================================================================================


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
import numpy as np
import h5py
from logging import getLogger
from uuid import uuid1
from pathlib import Path
import fasteners
from joblib import Parallel, delayed

DESCRIPTOR_CHOICES = ["aso", "aeif"]

OS_NCORES = os.cpu_count() // 2

arg_parser = ArgumentParser(
    "molli gbca",
    description="This module can be used for standalone computation of descriptors",
)

arg_parser.add_argument(
    "descriptor",
    choices=DESCRIPTOR_CHOICES,
    type=str.lower,
    help="This selects the specific descriptor to compute.",
)

arg_parser.add_argument(
    "library",
    metavar="CLIB_FILE",
    action="store",
    type=Path,
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

# arg_parser.add_argument(
#     "-c",
#     "--chunksize",
#     action="store",
#     type=int,
#     metavar=512,
#     default=512,
#     help="",
# )


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
    metavar="<grid.hdf5>",
    type=str,
    default=None,
    help="File that contains the information about the gridpoints.",
)

arg_parser.add_argument(
    "-o",
    "--output",
    action="store",
    metavar="<lib_aso.hdf5>",
    type=str,
    default=None,
    help="File that contains the information about the gridpoints.",
)

arg_parser.add_argument(
    "--dtype",
    type=np.dtype,
    default="float32",
    help="Specify the data format to be used for grid parameter storage.",
)

arg_parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite the existing descriptor file",
)


@delayed
def _aso_worker(
    _lib: ml.ConformerLibrary,
    _gfpath: str | Path,
    _output: str | Path,
    keys: list[str],
    weighted: bool = False,
    dtype: str | np.dtype = "float32",
):
    _lk_path = ml.aux.rwlock(_gfpath)
    lock = fasteners.InterProcessReaderWriterLock(_lk_path)

    with lock.read_lock():
        with h5py.File(_gfpath, mode="r") as h5f:
            grid = np.asarray(h5f["grid"])
            pruned = {k: np.asarray(h5f["grid_pruned_idx"][k]) for k in keys}

    with _lib.reading():
        ensembles = {k: _lib[k] for k in keys}

    asos = {}
    for k, ens in ensembles.items():
        aso = np.zeros(grid.shape[0], dtype=dtype)
        where = pruned[k]
        aso_pruned = ml.descriptor.aso(ens, grid[where], weighted=weighted)
        aso[where] = aso_pruned
        asos[k] = aso

    with lock.write_lock():
        with h5py.File(_output, mode="a") as of:
            for k, aso in asos.items():
                of[k] = aso


@delayed
def _aeif_worker(
    _lib: ml.ConformerLibrary,
    _gfpath: str | Path,
    _output: str | Path,
    keys: list[str],
    weighted: bool = False,
    dtype: str | np.dtype = "float32",
):
    _lk_path = ml.aux.rwlock(_gfpath)
    lock = fasteners.InterProcessReaderWriterLock(_lk_path)

    with lock.read_lock():
        with h5py.File(_gfpath, mode="r") as h5f:
            grid = np.asarray(h5f["grid"])
            pruned = {k: np.asarray(h5f["grid_pruned_idx"][k]) for k in keys}
            nearest = {k: np.asarray(h5f["nearest_atom_idx"][k]) for k in keys}

    with _lib.reading():
        ensembles = {k: _lib[k] for k in keys}

    aeifs = {}
    for k, ens in ensembles.items():
        aeif = np.zeros(grid.shape[0], dtype=dtype)
        where = pruned[k]
        aeif_pruned = ml.descriptor.aeif(
            ens, grid[where], nearest_atom_idx=nearest[k], weighted=weighted
        )
        aeif[where] = aeif_pruned
        aeifs[k] = aeif

    with lock.write_lock():
        with h5py.File(_output, mode="a") as of:
            for k, aeif in aeifs.items():
                of[k] = aeif


def molli_main(args, **kwargs):
    parsed = arg_parser.parse_args(args)
    logger = getLogger("molli.scripts.gbca")

    if parsed.output is None:
        out_name = Path(parsed.library).stem + f"_{parsed.descriptor}.hdf5"
        out_path = Path(parsed.library).parent / out_name
    else:
        out_path = Path(parsed.output)

    if parsed.grid is None:
        grid_name = Path(parsed.library).stem + "_grid.hdf5"
        grid_path = Path(parsed.library).parent / grid_name
    else:
        grid_path = Path(parsed.grid)

    parallel = Parallel(
        n_jobs=parsed.nprocs,
        backend="loky",
        return_as="generator",
    )

    library = ml.ConformerLibrary(parsed.library, readonly=True)

    with library.reading():
        if out_path.is_file():
            with h5py.File(out_path, mode="w" if parsed.overwrite else "r") as f:
                existing = set(f.keys())
            keys = sorted(library.keys() ^ existing)
        else:
            existing = None
            keys = sorted(library.keys())

    logger.info(
        f"To be computed: {len(keys)} ensembles. Skipping {len(library) - len(keys)}"
    )

    if len(keys) == 0:
        logger.info("Nothing to compute. Exiting.")
        exit(0)

    match parsed.descriptor:
        case "aso":
            for result in tqdm(
                parallel(
                    _aso_worker(
                        library,
                        grid_path,
                        out_path,
                        batch,
                        dtype=parsed.dtype,
                        weighted=parsed.weighted,
                    )
                    for batch in ml.aux.batched(keys, parsed.batchsize)
                ),
                desc=f"Computing descriptor ASO{' (weighted)' if parsed.weighted else ''}",
                total=ml.aux.len_batched(keys, parsed.batchsize),
            ):
                pass

        case "aeif":
            for result in tqdm(
                parallel(
                    _aeif_worker(
                        library,
                        grid_path,
                        out_path,
                        batch,
                        dtype=parsed.dtype,
                        weighted=parsed.weighted,
                    )
                    for batch in ml.aux.batched(keys, parsed.batchsize)
                ),
                desc=f"Computing descriptor AEIF{' (weighted)' if parsed.weighted else ''}",
                total=ml.aux.len_batched(keys, parsed.batchsize),
            ):
                pass

        case _:
            raise NotImplementedError

    # # Test if molli is launched via openmpi
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    # if size > 1:
    #     # This should only happen if MPI launch detected.
    #     if rank == 0:
    #         print(ml.config.SPLASH)
    #         logger.info(
    #             f"MPI launch detected: universe size {size}. --nprocs CMD parameter ignored even if specified."
    #         )
    #         logger.debug(f"Shared  lock file: {shared_lkfile}")
    #         logger.debug(f"Scratch lock file: {scratch_lkfile}")

    #     lib = ml.ConformerLibrary(parsed.conflib)
    #     grid = np.load(parsed.grid)

    #     # This enables strided access to the batch
    #     for batch_idx in range(rank, lib.n_batches(parsed.batchsize), size):
    #         try:
    #             with ml.ConformerLibrary.new(
    #                 ml.config.SCRATCH_DIR / f"molli-gbca-{batch_idx}-{rank}.mlib",
    #                 overwrite=True,
    #             ) as templib, lib:
    #                 pos = batch_idx * parsed.batchsize
    #                 chunksize = min(len(lib) - pos, parsed.batchsize)
    #                 # This creates a
    #                 lib.copy_chunk(templib, pos, chunksize)
    #         except Exception as xc:
    #             logger.critical("Could not copy batch {batch_idx}")
    #             logger.exception(xc)
    #             comm.Abort(1)
    #         logger.info(f"Batch {batch_idx} was copied into {templib.path}")
    #         with h5py.File(
    #             ml.config.SCRATCH_DIR / f"aso-{batch_idx}-{rank}.hdf5", "w"
    #         ) as f, templib:
    #             for key in templib.keys():
    #                 f[key] = ml.descriptor.chunky_aso(
    #                     templib[key], grid, parsed.chunksize
    #                 )

    #         logger.info(
    #             f"Computing ASO for batch {batch_idx} (last key: {key}) finished."
    #         )
    #     comm.barrier()
    #     if rank == 0:
    #         logger.info(f"Scattering the library across all workers is finished.")

    # else:
    #     # Using python multiprocessing now.
    #     logger.info(f"Using python multiprocessing with {parsed.nprocs} processes.")

    # # print(
    # #     f"Will compute descriptor {'w' if parsed.weighted else ''}{parsed.descriptor} using {parsed.nprocs} cores."
    # # )
    # #
