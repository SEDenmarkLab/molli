# this file will handle optional post processing for that data with post_processing.py (option to modify in place or create new),
# run tsne/PCA on the data and generate plots with customizability functionality
from . import helpers
from . import post_processing as pc
from pathlib import Path
import pandas as pd
import json
import warnings
from kneed import KneeLocator


def tsne_processing(
    input: pd.DataFrame | str | Path,
    output: str | Path,
    perplex: int = 5,
    upper_k: int = 20,
    plot: bool = False,
):  # -> tuple
    """
    intakes aso.h5 file or dataframe, converts to dataframe, does kmeans clustering and automatically selects
    optimal number of clusters. Then generates tsne values, and optionally writes data as JSON dataframe to given output filepath.
    format of output is output + '_exemplars.json' and output + '_values_and_clusters.json'
    optional parameter for perplexity of tsne, defaults to 5, and inclusive upper bound of k clusters, defaults to 20.
    returns data as dataframe in following format:
    catalyst_id as string | tSNE1 value as float | tSNE2 value as float | (1 -> k columns of cluster assignments)
    and a list of lists of exemplar catalysts. Saves plot as a .png to file path if plot=True
    """
    import pandas as pd

    if not isinstance(input, pd.DataFrame):
        try:
            temp = pc.unpack_h5py(input)
            input = temp
        except Exception as exp:
            warnings.warn(f"Invalid filepath: {exp!s}")

    df = helpers.tsne_score(input, perplexity=perplex, save_path=output, plot=plot)

    all_exemplars = []
    distortions = []
    for i in range(1, upper_k + 1):
        kmeans = helpers.get_kmeans(input, i)
        exemplars = helpers.exemplar_selector(kmeans, input)
        all_exemplars.append(exemplars)
        df["cluster assignments (k=" + str(i) + ")"] = helpers.assignments(exemplars, input)
        _, distort = helpers.distortion_calculation(input, pd.DataFrame(kmeans.cluster_centers_))
        distortions.append(distort)

    kl = KneeLocator(
        list(range(1, upper_k + 1)),
        distortions,
        curve="convex",
        direction="decreasing",
        online=True,
    )
    
    processed_json(output, df, all_exemplars, kl.knee)


#    return df, all_exemplars


def pca_processing(
    input: pd.DataFrame | str | Path, output: str | Path, upper_k: int = 20, plot: bool = False
):  # -> tuple
    """
    Same as tsne_processing, but outputs PCA scores instead of tsne scores. Also lacks a perplexity parameter.
    """
    if not isinstance(input, pd.DataFrame):
        try:
            temp = pc.unpack_h5py(input)
            input = temp
        except Exception as exp:
            warnings.warn(f"Invalid filepath: {exp!s}")

    df = helpers.pca_score(input, save_path=output, plot=plot)

    all_exemplars = []
    distortions = []
    for i in range(1, upper_k + 1):
        kmeans = helpers.get_kmeans(input, i)
        exemplars = helpers.exemplar_selector(kmeans, input)
        all_exemplars.append(exemplars)
        df["cluster assignments (k=" + str(i) + ")"] = helpers.assignments(exemplars, input)
        _, distort = helpers.distortion_calculation(input, pd.DataFrame(kmeans.cluster_centers_))  # for knee calculation
        distortions.append(distort)

    kl = KneeLocator(
        list(range(1, upper_k + 1)),
        distortions,
        curve="convex",
        direction="decreasing",
        online=True,
    )

    processed_json(output, df, all_exemplars, kl.knee)


#    return df, all_exemplars


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
