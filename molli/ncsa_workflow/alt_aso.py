from concurrent.futures import ThreadPoolExecutor
import molli as ml
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Value
import numpy as np
import h5py
from pathlib import Path
import warnings
# import os
# This is a bypass for DASK (to be abandoned in future molli development)

# lib = ml.ConformerLibrary("../../../out_conformers1/conformers-cmd-test2-nolink.mlib")
# grid = np.load("../molli/lib_gen/test_aso/grid.npy")
# grid = np.load("grid.npy")

# print("grid shape:", grid.shape)

# BATCH_SIZE = 256

print("Allocating storage for descriptors")

# defaults for n_proc and batch_size given in parsing script gbca
def aso_description(clib:ml.ConformerLibrary | str | Path, grid: np.ndarray, output: str | Path | None, n_proc: int, batch_size: int, chunk_size: int):
    '''
    Given a conformer library, as either the object or file path to it, will calculate aso descriptors for each catalyst
    and write as an h5 file in output path. If output not passed, will write as 'aso.h5' in same directory
    '''
    try:
        if isinstance(clib, (str, Path)):
            lib = ml.ConformerLibrary(clib)
        else:
            lib = clib
    except Exception as exp:
        warnings.warn(f'Invalid ConformerLibrary: {exp!s}')

    if output is None:
        output = 'aso.h5'

    try:
        
        for batch in tqdm(lib.yield_in_batches(batch_size), dynamic_ncols=True, position=0, desc="Loading batches of conformers", total=len(lib)//batch_size):
            futures = []
            with ThreadPoolExecutor(max_workers=n_proc) as pool:
                    # data = np.empty((len(batch), grid.shape[0]), dtype="float32")
                    for ens in tqdm(batch, dynamic_ncols=True, position=1, leave=False, desc="Submitting calculations"):
                        fut = pool.submit(ml.descriptor.filtered_parallel_aso, ens, grid, n_threads=16, chunksize=chunk_size) # n_proc for threadpoolexecutor or here?
                        futures.append(fut)
            with h5py.File(output, "a") as output:     
                for i, f in enumerate(tqdm(futures, dynamic_ncols=True, position=2, leave=False, desc="Gathering calculation results")):
                    output.create_dataset(batch[i].name, data=f.result(), dtype="f4")
    except Exception as exp:
        warnings.warn(f'Invalid filepath: {exp!s}')