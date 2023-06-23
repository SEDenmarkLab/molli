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


def processed_json(
        output: str | Path, df: pd.DataFrame, all_exemplars: list, distortions: int, knee: int, mode: str
):
    '''
    Outputs given inputs as a dictionary in json to output filepath.
    '''
    try:
        result = {
            "mode": mode,
            "coordinates": df.iloc[:, 0:2].to_dict(),
            "clusterAssignments": df.iloc[:, 2:].to_dict(),
            "defaultNumberOfClusters": int(knee),
            "distortions": distortions,
            "exemplars": all_exemplars
        }

        with open(output, 'w') as f:
            f.write(json.dumps(result, indent=4)) 
    except Exception as exp:
        warnings.warn(f"Error with output file creation: {exp!s}")


'''
def processed_json( # END OF DISTORTIONS LIST IS INTEGER CORRESPONDING TO KNEE
    output: str | Path, df: pd.DataFrame, all_exemplars: list, distortions: int, knee: int
):  # given output filepath, will create files of format: (output + '_exemplars.json')
    try:  #  or (output + '_values_and_clusters.json')
        distortions.append(int(knee))
        with open((output + "_distortions.json"), "w") as f:
            f.write(json.dumps(distortions))
        with open((output + "_exemplars.json"), "w") as f:
            f.write(json.dumps(all_exemplars))
        df.to_json(output + "_values_and_clusters.json")
    except Exception as exp:
        warnings.warn(f"Error with output file creation: {exp!s}")  # move to argument parser?
'''

def unpack_json(input: str | Path) -> tuple:   # reverses processed_json
    try:                         # outputs tuple of (values/clusters dataframe, exemplars list, distortions)
        with open(input, 'r') as f:
            dict = json.load(f)

        mode = dict.get("mode")
        values = dict.get("coordinates")
        assignments = dict.get("clusterAssignments")
        knee = dict.get("defaultNumberOfClusters")
        distortions = dict.get("distortions")
        exemplars = dict.get("exemplars")

    except Exception as exp:
        warnings.warn(f"Error with reading in JSON files: {exp!s}")

    return values, assignments, exemplars, knee, mode, distortions
