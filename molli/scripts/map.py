# ================================================================================
# This file is part of `molli 1.0`
# (https://github.com/SEDenmarkLab/molli)
#
# Developed by Alexander S. Shved <shvedalx@illinois.edu>
#
# S. E. Denmark Laboratory, University of Illinois, Urbana-Champaign
# https://denmarkgroup.illinois.edu/
#
# Copyright 2022-2024 The Board of Trustees of the University of Illinois.
# All Rights Reserved.
#
# Licensed under the terms MIT License
# The License is included in the distribution as LICENSE file.
# You may not use this file except in compliance with the License.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# ================================================================================


"""
`molli map` script maps an arbitrarily defined python script with parallelization
"""

from functools import partial
from argparse import ArgumentParser
import molli as ml
from molli.external import openbabel
from itertools import (
    permutations,
    combinations_with_replacement,
    combinations,
    repeat,
    chain,
    product,
)
from typing import Callable
from tqdm import tqdm
import multiprocessing as mp
import os
from math import comb, perm, factorial
from joblib import Parallel, delayed
from molli._aux.mpi import detect_mpi_launch
from tempfile import TemporaryDirectory
from pathlib import Path

OS_NCORES = os.cpu_count() // 2

arg_parser = ArgumentParser(
    "molli combine",
    description="Read a molli library and perform some basic inspections",
)

arg_parser.add_argument(
    "script",
    help="This is a python file that defines a molli_main function",
)

arg_parser.add_argument(
    "-t",
    "--target",
    action="store",
    metavar="<lib>",
    help="Target library that the function is going to be applied to",
    required=True,
)

arg_parser.add_argument(
    "-n",
    "--nprocs",
    action="store",
    metavar=1,
    default=1,
    type=int,
    help="Number of processes to be used in parallel",
)

arg_parser.add_argument(
    "-b",
    "--batchsize",
    action="store",
    metavar=1,
    default=1,
    type=int,
    help="Number of molecules to be processed at a time on a single core",
)

arg_parser.add_argument(
    "-o",
    "--output",
    action="store",
    metavar="<combined.mlib>",
    help="Output library",
)

arg_parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite the target files if they exist (default is false)",
    default=False,
)


@delayed
def _runner(fx, items):
    return [(k, fx(v)) for k, v in items]


def molli_main(args, **kwargs):
    parsed = arg_parser.parse_args(args)

    # Now loading an external python code
    # This will enable to do _ext_module.molli_main(in, out)
    _ext_module = ml.aux.load_external_module(parsed.script, "_ext_module")

    assert issubclass(_ext_module.IN_CTYPE, ml.storage.Collection)
    IN_CTYPE: ml.storage.Collection = _ext_module.IN_CTYPE

    if hasattr(_ext_module, "OUT_CTYPE"):
        if _ext_module.OUT_CTYPE is None:
            assert issubclass(_ext_module.OUT_CTYPE, ml.storage.Collection)

    OUT_CTYPE: ml.storage.Collection = getattr(_ext_module, "OUT_CTYPE", None)

    source = IN_CTYPE(parsed.target, readonly=True)

    if OUT_CTYPE is not None:
        destination = OUT_CTYPE(
            parsed.output,
            readonly=False,
            overwrite=parsed.overwrite,
        )

    # This is the name of the scratch directory, if needed

    if detect_mpi_launch():
        # This should occur only if molli was called like `mpirun -n 4 molli map ...`
        # and therefore we will need a different mode or parallelization
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if comm.Get_rank() == 0:
            if parsed.nprocs > 1:
                print(
                    "Using molli map with mpiexec or -n parameter are mutually exclusive."
                )
                comm.Abort(1)

            print(f"Detected an mpi launch with {size=}")

            with source.reading():
                keys = sorted(source.keys())

            keys_split = [keys[s::size] for s in range(size)]
        else:
            keys_split = 0

        keys_split = comm.scatter(keys_split, root=0)

        # Each process will be writing in its own output file
        # At the end all outputs will be concatenated

        with TemporaryDirectory(dir=ml.config.SCRATCH_DIR) as td:
            # Create a temporary folder in the SCRATCH
            temp_fname = Path(td) / Path(parsed.output).name + f".{rank:0>4}"
            if OUT_CTYPE is not None:
                dest_temp = OUT_CTYPE(
                    temp_fname,
                    readonly=False,
                    overwrite=parsed.overwrite,
                )

            with dest_temp.writing(), source.reading():
                for k in keys_split:
                    res = _ext_module.main(source[k])
                    dest_temp[k] = res

    else:

        parallel = Parallel(n_jobs=parsed.nprocs, return_as="generator")

        with source.reading(), destination.reading():
            keys_tbd = source.keys() ^ destination.keys()

        with source.reading():
            # error_counts = 0
            for results in (
                pb := tqdm(
                    parallel(
                        (
                            _runner(_ext_module.main, b)
                            for b in ml.aux.batched(
                                ((k, source[k]) for k in keys_tbd), parsed.batchsize
                            )
                        )
                    ),
                    total=ml.aux.len_batched(keys_tbd, parsed.batchsize),
                    dynamic_ncols=True,
                )
            ):
                with destination.writing():
                    for k, v in results:
                        if isinstance(v, Exception):
                            pb.write(f"Error while processing {k}: {v}")
                        else:
                            destination[k] = v
