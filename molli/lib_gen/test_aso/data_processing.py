# this file will handle optional post processing for that data with post_processing.py (option to modify in place or create new),
# run tsne/PCA on the data and generate plots with customizability functionality,
# and command line argument parsing for all of the above
import sys
import molli.lib_gen.test_aso.helpers as helpers
import molli.lib_gen.test_aso.post_processing as pc         # FIX THESE !
import molli as ml
import argparse
from pathlib import Path
import pandas as pd
import json
import warnings

def tsne_processing(input: pd.DataFrame | str | Path,  output: str | Path, perplex: int=5, upper_k: int=20, plot: bool=False) -> tuple:
    '''
    skips plotting, simply intakes aso.h5 file or dataframe, converts to dataframe, does kmeans clustering and automatically selects
    optimal number of clusters. Then generates tsne values, and optionally writes data as JSON dataframe to given output filepath.
    format of output is output + '_exemplars.json' and output + '_values_and_clusters.json'
    optional parameter for perplexity of tsne, defaults to 5, and inclusive upper bound of k clusters, defaults to 20.
    returns data as dataframe in following format:
    catalyst_id as string | tSNE1 value as float | tSNE2 value as float | (1 -> k columns of cluster assignments)
    and a list of lists of exemplar catalysts
    '''
    if not isinstance(input, pd.DataFrame):
        try:
            temp = pc.unpack_h5py(input)
            input = temp
        except Exception as exp:
            print(f'Invalid filepath: {exp!s}')
            return
        
    df = helpers.tsne_score(input, perplexity=perplex, save_path=output, plot=plot)

    all_exemplars = []
    for i in range(1, upper_k + 1):
        kmeans = helpers.get_kmeans(input, i)
        exemplars = helpers.exemplar_selector(kmeans, input)
        all_exemplars.append(exemplars)
        df['cluster assignments (k=' + str(i) +')'] = helpers.assignments(exemplars, input)

    processed_json(output, df, all_exemplars)

    return df, all_exemplars


def pca_processing(input: pd.DataFrame | str | Path,  output: str | Path, upper_k: int=20, plot: bool=False) -> tuple:
    if not isinstance(input, pd.DataFrame):
        try:
            temp = pc.unpack_h5py(input)
            input = temp
        except Exception as exp:
            print(f'Invalid filepath: {exp!s}')
            return
        
    df = helpers.pca_score(input, save_path=output, plot=plot)

    all_exemplars = []
    for i in range(1, upper_k + 1):
        kmeans = helpers.get_kmeans(input, i)
        exemplars = helpers.exemplar_selector(kmeans, input)
        all_exemplars.append(exemplars.index.to_list())
        df['cluster assignments (k=' + str(i) +')'] = helpers.assignments(exemplars, input)

    processed_json(output, df, all_exemplars)

    return df, all_exemplars


def processed_json(output: str | Path, df: pd.DataFrame, all_exemplars: list):  # given output filepath, will create files of format: (output + '_exemplars.json')
        try:                                                                    #  or (output + '_values_and_clusters.json')
            with open((output + '_exemplars.json'), 'w') as f:
                f.write(json.dumps(all_exemplars))
            df.to_json(output + '_values_and_clusters.json')
        except Exception as exp:
            warnings.warn(f'Error with output file creation: {exp!s}')  # move to argument parser?