import pandas as pd
import numpy as np # REMOVE IF I MOVE DISTORTIONS
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator


def get_kmeans(df: pd.DataFrame, n_clusters: int) -> KMeans:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # suppresses future warning
    kmeans.fit(
        df
    )  # intel openmp issue occurs here (make sure to install all packages into same env)
    return kmeans


# generates kmeans elbow plot, with red line corresponding to kneed calculation of knee location
# returns a tuple of list of distortion values per number of clusters, and integer corresponding to knee location
def kmeans_elbow(max_clusters: int, distortions: list) -> int:
    max_clusters += 1  # makes maximum inclusive

    kl = KneeLocator(
        list(range(1, max_clusters + 1)),
        distortions,
        curve="convex",
        direction="decreasing",
        online=True,
    )

    return kl.knee


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