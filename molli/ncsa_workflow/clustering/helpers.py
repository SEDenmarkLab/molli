#import os
#os.environ['OMP_NUM_THREADS'] = '8'

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from pathlib import Path
import warnings
import json



# reads in the h5 file in a nice way for pandas
def unpack_h5py(file):
    df = {}
    with h5py.File(file, 'r') as f:
        # print(f.keys())
        # exit()
        for x in tqdm(f.keys()):
            df[x] =np.array(f[x])
    return pd.DataFrame.from_dict(df).transpose()


def processed_json( # END OF EXEMPLARS LIST IS INTEGER CORRESPONDING TO KNEE
    output: str | Path, df: pd.DataFrame, all_exemplars: list, knee: int
):  # given output filepath, will create files of format: (output + '_exemplars.json')
    try:  #  or (output + '_values_and_clusters.json')
        all_exemplars.append(int(knee))
        with open((output + "_exemplars.json"), "w") as f:
            f.write(json.dumps(all_exemplars))
        df.to_json(output + "_values_and_clusters.json")
    except Exception as exp:
        warnings.warn(f"Error with output file creation: {exp!s}")  # move to argument parser?


def unpack_json(input_values: str, input_exemplars: str) -> tuple:   # reverses processed_json
    try:                                                             # outputs tuple of (values/clusters dataframe, exemplars list)
        with open(input_values, 'r') as values, open(input_exemplars, 'r') as exemplars:
            df = pd.read_json(values, convert_axes=False)
            exemp = json.load(exemplars)
    except Exception as exp:
        warnings.warn(f"Error with reading in JSON files: {exp!s}")

    return df, exemp
