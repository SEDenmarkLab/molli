from concurrent.futures import ThreadPoolExecutor
import molli as ml
from tqdm import tqdm
from multiprocessing import Value   # for tqdm
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
        global loading, submitting, gathering
        loading = Value("Q", 0, lock=True)
        submitting = Value("Q", 0, lock=True)
        gathering = Value("Q", 0, lock=True)
        cur_load = 0
        with tqdm(lib.yield_in_batches(batch_size), dynamic_ncols=True, position=0, desc="Loading batches of conformers", total= -(len(lib)//-batch_size)) as lb: # upsidedown floor div
            for batch in lb:
                futures = []
                cur_sub = 0
                cur_gath = 0
                with ThreadPoolExecutor(max_workers=n_proc) as pool:
                        # data = np.empty((len(batch), grid.shape[0]), dtype="float32")
                        with tqdm(batch, dynamic_ncols=True, position=1, leave=False, desc="Submitting calculations") as sb:
                            for ens in sb:
                                fut = pool.submit(ml.descriptor.filtered_parallel_aso, ens, grid, n_threads=16, chunksize=chunk_size) # n_proc for threadpoolexecutor or here?
                                futures.append(fut)
                                with submitting.get_lock():
                                    sb.update(submitting.value - cur_sub)
                                    cur_sub = submitting.value
                                
                with h5py.File(output, "a") as out:     
                    with tqdm(futures, dynamic_ncols=True, position=2, leave=False, desc="Gathering calculation results") as gb:
                        for i, f in enumerate(gb):
                            out.create_dataset(batch[i].name, data=f.result(), dtype="f4")
                            with gathering.get_lock():
                                gb.update(gathering.value - cur_gath)
                                cur_gath = gathering.value

                with loading.get_lock():
                    lb.update(loading.value - cur_load)
                    cur_load = loading.value

    except Exception as exp:
        warnings.warn(f'Invalid filepath: {exp!s}')
