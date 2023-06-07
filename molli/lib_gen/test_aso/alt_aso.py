from concurrent.futures import ThreadPoolExecutor
import molli as ml
from tqdm import tqdm
import numpy as np
import h5py
from pathlib import Path
# This is a bypass for DASK (to be abandoned in future molli development)

#lib = ml.ConformerLibrary("../../../out_conformers1/conformers-cmd-test2-nolink.mlib")
grid = np.load("molli/lib_gen/test_aso/grid.npy")      # ISSUE WITH FINDING THIS FILE NOW

print("grid shape:", grid.shape)

BATCH_SIZE = 256

print("Allocating storage for descriptors")

def aso_description(clib:ml.ConformerLibrary | str | Path, output: str | Path | None ):
    '''
    Given a conformer library, as either the object or file path to it, will calculate aso descriptors for each catalyst
    and return a pandas DataFrame. Will also write 
    '''
    try:
        if isinstance(clib, (str, Path)):
            lib = ml.ConformerLibrary(clib)
        else:
            lib = clib
    except Exception as exp:
        print(f'Invalid ConformerLibrary: {exp!s}')
        return

    if output is None:
        output = 'aso.h5'

    try:
        with ThreadPoolExecutor(max_workers=64) as pool, h5py.File(output + '_aso.h5', "w") as output:
            for batch in tqdm(lib.yield_in_batches(BATCH_SIZE), dynamic_ncols=True, position=0, desc="Loading batches of conformers", total=len(lib)//BATCH_SIZE):
                futures = []
                data = np.empty((len(batch), grid.shape[0]), dtype="float32")
                for ens in tqdm(batch, dynamic_ncols=True, position=1, leave=False, desc="Submitting calculations"):
                    fut = pool.submit(ml.descriptor.filtered_parallel_aso, ens, grid, n_threads=16, chunksize=1024)
                    futures.append(fut)
                
                for i, f in enumerate(tqdm(futures, dynamic_ncols=True, position=2, leave=False, desc="Gathering calculation results")):
                    output.create_dataset(batch[i].name, data=f.result(), dtype="f4")
    except Exception as exp:
        print(f'Invalid filepath: {exp!s}')
        return


#if __name__ == "__main__":

#    print('Success!')