import os

os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import scipy.stats as st
import h5py
from tqdm import tqdm

import statsmodels.graphics.gofplots as sm
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from kneed import KneeLocator


# subsets is a dict. The keys are labels for the data subset. The value is a 3-tuple. The first element is a subset of index
# labels that should appear in full_df. The second is the alpha value in pyplot to plot these with. The third is the pyplot color.
#  Returns the transformed data.
def tsne_score(
    full_df: pd.DataFrame,
    dimensions=2,
    name="test",
    plot=False,
    print_tsne_values=False,
    save_path=None,
    perplexity=30,
    highlight_df: pd.DataFrame = None,
) -> pd.DataFrame:  # clean up later
    # this will allows us to pass in the TSNE solution on subsequent calls to this function, if we want
    if len(full_df.columns) != 2:
        full_df = pd.DataFrame(
            TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(full_df),
            index=full_df.index,
        )
    else:
        print("This data already had 2 dimensions! Skipping tSNE.")

    if plot:
        tsne_plot(full_df, highlight_df, dimensions, name, save_path)   # REMOVE THIS

    return full_df


# subsets give label:list of catalyst handles value pairs for each subset
# return transformed dataset
def pca_score(
    full_df: pd.DataFrame,
    dimensions=2,
    name="test",
    save_path=None,
    plot=False,
    highlight_df: pd.DataFrame = None,
) -> pd.DataFrame:
    _pca_test = PCA(n_components=2, random_state=42).fit(full_df)
    full_df = pd.DataFrame(_pca_test.transform(full_df), index=full_df.index)

    if plot:
        pca_plot(full_df, highlight_df, dimensions, name, save_path)

    return full_df


# returns distortion for kmeans analysis of dataframe given dataframe of cluster centroids
def distortion_calculation(space: pd.DataFrame, sample: pd.DataFrame):
    # separate out all of the other samples from our current sample
    #    rest = space.loc[[i for i in space.index if i not in sample.index], :]
    # print(space.shape)
    # print(rest.shape)
    # this function gives https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin_min.html
    assignments, distances = pairwise_distances_argmin_min(
        space, sample
    )  # assignments is ndarray of indexes for space, s.t. y[assignments[i]] is closest to sample[i]
    # print(cluster_assignments, distances)
    # print(len(cluster_assignments))
    # exit()
    # calculate distortions
    distortion = np.sum([i**2 for i in distances])

    return assignments, distortion


# generates kmeans elbow plot, with red line corresponding to kneed calculation of knee location
# returns a tuple of list of distortion values per number of clusters, and integer corresponding to knee location
def kmeans_elbow(
    df: pd.DataFrame, max_clusters=20, name="test", save_path=None, plot=False
) -> tuple:
    max_clusters += 1  # makes maximum inclusive

    distortions = []
    for k in range(1, max_clusters):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # suppresses future warning
        kmeans.fit(df)
        _, distort = distortion_calculation(df, pd.DataFrame(kmeans.cluster_centers_))
        distortions.append(distort)

    kl = KneeLocator(
        list(range(1, max_clusters)),
        distortions,
        curve="convex",
        direction="decreasing",
        online=True,
    )

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True, axis="x", linestyle="--")

        plt.title(f"kmeans elbow plot (distortion): {name}")
        plt.xlabel("k clusters")
        plt.xticks(range(1, max_clusters))
        plt.ylabel("distortion")

        ax.plot(range(1, max_clusters), distortions)
        plt.axvline(kl.knee, 0, 1, color="r")
        plt.text(kl.knee + 0.1, 0.5, f"knee is {kl.knee}")

        if save_path:
            fig.savefig(save_path + f"kmeans elbow {name}.png", bbox_inches="tight")

    return distortions, kl.knee


# returns dataframe containing num_cluster number of indices, each corresponding to the catalyst closest to the cluster centroid
def exemplar_selector(kmeans: KMeans, data: pd.DataFrame) -> list:
    closest, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, data
    )  # closest is ndarray of indexes s.t. data[closest[i]] is row closest to clusters[i]
    exemplars = data.iloc[closest, :]
    return exemplars.index.to_list()


# returns dataframe containing each index in data, each corresponding to exemplar that it is closest to (its cluster group)
def assignments(exemplars: list, data: pd.DataFrame) -> pd.DataFrame:
    closest, _ = pairwise_distances_argmin_min(
        data, data.loc[exemplars]
    )  # reverse of exemplars, array of indexes of clusters s.t. exemplars[closest[i]] is exemplar closest to data[i]
    assignments = pd.DataFrame(index=data.index)  # change to actual exemplars
    assignments["cluster assignments"] = closest
    return assignments


def sorting_legend(label):
    x, y, _ = map(int, label.split("_")) # works for labels of format "X_Y_Y", needs to be changed for different implementations
    return x, y


def tsne_plot(  # functionality to be changed. Meant to be called on fully processed tsne dataframe, with all cluster assignment columns
    t_df: pd.DataFrame, highlight: list, dimensions: int, name: str, save_path, num_clusters: int
):
    fig = plt.figure()
    # make 2D projection
    ax = fig.add_subplot(111)

    ax.grid(False)
    plt.xlabel("tSNE1")
    plt.ylabel("tSNE2")
    plt.title(f"tSNE {dimensions} components: {name}")

    # now coloring based on cluster, maximum 20 color variations
    # if this vizualization becomes more important, implement better coloration functionality
    c = [
        'mediumturquoise',
        'firebrick',
        'purple',
        'orange',
        'darkcyan',
        'red',
        'fuchsia',
        'lawngreen',
        'deeppink',
        'saddlebrown',
        'black',
        'lightcoral',
        'dodgerblue',
        'grey',
        'forestgreen',
        'navy',
        'gold',
        'darkgreen',
        'crimson',
        'yellow',
    ]

    # cluster assignment number = color list index
    if highlight is None:
        for i in t_df.index.to_list():
            temp = t_df.loc[i]
            ax.scatter(
                temp[0], temp[1], alpha=1, color=c[int(temp[num_clusters + 1])], label=i
            )
    else:
        for (
            i
        ) in t_df.index.to_list():  # for increasing opacity of specific catalysts (like exemplars)
            if i in highlight:
                a = 1
                style = "*"
                zorder = 2
            else:
                a = 0.75
                style = "o"
                zorder = 1

            temp = t_df.loc[i]
            ax.scatter(
                temp[0],
                temp[1],
                alpha=a,
                marker=style,
                color=c[int(temp[num_clusters + 1])],
                label=i,
                zorder=zorder
            )

    # Can adjust bbox_to_anchor to move legend
    handles, labels = ax.get_legend_handles_labels()
    sorted_labels, sorted_handles = zip(
        *sorted(zip(labels, handles), key=lambda x: sorting_legend(x[0]))
    )
    # legend specific to the 15 substituent format, will need to change to be more general
    ax.legend(sorted_handles, sorted_labels, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=5)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def pca_plot(
    t_df: pd.DataFrame,
    highlight: list,
    dimensions: int,
    name: str,
    save_path,
    num_clusters: int
):
    fig = plt.figure()
    # make PCA projection for 2D
    ax = fig.add_subplot(111)

    ax.grid(False)
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title(f"PCA {dimensions}D: {name}")

    # print(f"Explained variance in PCA plot: {np.sum(pca_test.explained_variance_ratio_)}")

    # plot the whole in silico library
    c = [
        'mediumturquoise',
        'firebrick',
        'purple',
        'orange',
        'darkcyan',
        'red',
        'fuchsia',
        'lawngreen',
        'deeppink',
        'saddlebrown',
        'black',
        'lightcoral',
        'dodgerblue',
        'grey',
        'forestgreen',
        'navy',
        'gold',
        'darkgreen',
        'crimson',
        'yellow',
    ]

    if highlight is None:
        for i in t_df.index.to_list():
            temp = t_df.loc[i]
            ax.scatter(
                temp[0], temp[1], alpha=1, color=c[int(temp[num_clusters + 1])], label=i
            )
    else:
        for (
            i
        ) in t_df.index.to_list():  # for increasing opacity of specific catalysts (like exemplars)
            if i in highlight:
                a = 1
                style = "*"
                zorder = 2
            else:
                a = 0.75
                style = "o"
                zorder = 1

            temp = t_df.loc[i]
            ax.scatter(
                temp[0],
                temp[1],
                alpha=a,
                marker=style,
                color=c[int(temp[num_clusters + 1])],
                label=i,
                zorder=zorder
            )

    # Can adjust bbox_to_anchor to move legend
    handles, labels = ax.get_legend_handles_labels()
    sorted_labels, sorted_handles = zip(
        *sorted(zip(labels, handles), key=lambda x: sorting_legend(x[0]))
    )
    # legend specific to the 15 substituent format, will need to change to be more general
    ax.legend(sorted_handles, sorted_labels, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=5)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def get_kmeans(df: pd.DataFrame, n_clusters: int) -> KMeans:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # suppresses future warning
    kmeans.fit(
        df
    )  # intel openmp issue occurs here (make sure to install all packages into same env)
    return kmeans
