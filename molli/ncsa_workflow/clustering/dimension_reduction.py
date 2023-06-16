# this file will handle optional post processing for that data with post_processing.py (option to modify in place or create new),
# run tsne/PCA on the data and generate plots with customizability functionality
from . import helpers
from . import k_means
from pathlib import Path
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings


def tsne_processing(
    input: pd.DataFrame | str | Path,
    output: str | Path,
    perplex: int = 5,
    upper_k: int = 20,
):
    """
    intakes aso.h5 file or dataframe, converts to dataframe, does kmeans clustering and automatically selects
    optimal number of clusters. Then generates tsne values, and optionally writes data as JSON dataframe to given output filepath.
    format of output is output + '_exemplars.json' and output + '_values_and_clusters.json'
    optional parameter for perplexity of tsne, defaults to 5, and inclusive upper bound of k clusters, defaults to 20.
    returns data as dataframe in following format:
    catalyst_id as string | tSNE1 value as float | tSNE2 value as float | (1 -> k columns of cluster assignments)
    and a list of lists of exemplar catalysts. Saves plot as a .png to file path if plot=True
    """

    if not isinstance(input, pd.DataFrame):
        try:
            temp = helpers.unpack_h5py(input)
            input = temp
        except Exception as exp:
            warnings.warn(f"Invalid filepath: {exp!s}")

    df = tsne_score(input, perplexity=perplex, save_path=output)

    all_exemplars = []
    distortions = []
    for i in range(1, upper_k + 1):
        kmeans = k_means.get_kmeans(input, i)
        exemplars = k_means.exemplar_selector(kmeans, input)
        all_exemplars.append(exemplars)
        df["cluster assignments (k=" + str(i) + ")"] = k_means.assignments(exemplars, input)
        _, distort = k_means.distortion_calculation(input, pd.DataFrame(kmeans.cluster_centers_)) # for knee calculation
        distortions.append(distort)

    knee = k_means.kmeans_elbow(upper_k, distortions)
    
    helpers.processed_json(output, df, all_exemplars, knee)


def pca_processing(
    input: pd.DataFrame | str | Path, output: str | Path, upper_k: int = 20
):
    """
    Same as tsne_processing, but outputs PCA scores instead of tsne scores. Also lacks a perplexity parameter.
    """
    if not isinstance(input, pd.DataFrame):
        try:
            temp = helpers.unpack_h5py(input)
            input = temp
        except Exception as exp:
            warnings.warn(f"Invalid filepath: {exp!s}")

    df = pca_score(input, save_path=output)

    all_exemplars = []
    distortions = []
    for i in range(1, upper_k + 1):
        kmeans = k_means.get_kmeans(input, i)
        exemplars = k_means.exemplar_selector(kmeans, input)
        all_exemplars.append(exemplars)
        df["cluster assignments (k=" + str(i) + ")"] = k_means.assignments(exemplars, input)
        _, distort = k_means.distortion_calculation(input, pd.DataFrame(kmeans.cluster_centers_))  # for knee calculation
        distortions.append(distort)

    knee = k_means.kmeans_elbow(upper_k, distortions)

    helpers.processed_json(output, df, all_exemplars, knee)


# subsets is a dict. The keys are labels for the data subset. The value is a 3-tuple. The first element is a subset of index
# labels that should appear in full_df. The second is the alpha value in pyplot to plot these with. The third is the pyplot color.
#  Returns the transformed data.
def tsne_score(
    full_df: pd.DataFrame,
    dimensions=2,
    name="test",
    perplexity=30,
) -> pd.DataFrame:  # clean up later
    # this will allows us to pass in the TSNE solution on subsequent calls to this function, if we want
    if len(full_df.columns) != 2:
        full_df = pd.DataFrame(
            TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(full_df),
            index=full_df.index,
        )
    else:
        print("This data already had 2 dimensions! Skipping tSNE.")

    return full_df


# subsets give label:list of catalyst handles value pairs for each subset
# return transformed dataset
def pca_score(
    full_df: pd.DataFrame,
    dimensions=2,
    name="test",
) -> pd.DataFrame:
    _pca_test = PCA(n_components=2, random_state=42).fit(full_df)
    full_df = pd.DataFrame(_pca_test.transform(full_df), index=full_df.index)

    return full_df
