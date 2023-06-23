import pandas as pd
from matplotlib import pyplot as plt



def sorting_legend(label):
    x, y, _ = map(int, label.split("_")) # works for labels of format "X_Y_Y", needs to be changed for different implementations
    return x, y


def tsne_plot(  # functionality to be changed. Meant to be called on fully processed tsne dataframe, with all cluster assignment columns
    scores: dict, assignments: dict, highlight: list, dimensions: int, name: str, save_path, num_clusters: int
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
        for i in scores['0']:
            color_index = int(assignments[i])
            ax.scatter(
                scores['0'][i], scores['1'][i], alpha=1, color=c[color_index], label=i
            )
    else:
        for i in scores['0']:  # for increasing opacity of specific catalysts (like exemplars)
            if i in highlight:
                a = 1
                style = "*"
                zorder = 2
            else:
                a = 0.75
                style = "o"
                zorder = 1

            color_index = int(assignments[i])
            ax.scatter(
                scores['0'][i], scores['1'][i], alpha=a, marker=style, color=c[color_index], label=i, zorder=zorder
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
    scores: dict,
    assignments: dict,
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
        for i in scores['0']:
            color_index = int(assignments[i])
            ax.scatter(
                scores['0'][i], scores['1'][i], alpha=1, color=c[color_index], label=i
            )
    else:
        for i in scores['0']:  # for increasing opacity of specific catalysts (like exemplars)
            if i in highlight:
                a = 1
                style = "*"
                zorder = 2
            else:
                a = 0.75
                style = "o"
                zorder = 1

            color_index = int(assignments[i])
            ax.scatter(
                scores['0'][i], scores['1'][i], alpha=a, marker=style, color=c[color_index], label=i, zorder=zorder
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


def elbow_plot(distortions: list, elbow: int, save_path=None, name: str = 'test'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True, axis="x", linestyle="--")

    plt.title(f"kmeans elbow plot (distortion): {name}")
    plt.xlabel("k clusters")
    plt.xticks(range(1, len(distortions)))
    plt.ylabel("distortion")

    ax.plot(range(1, len(distortions)), distortions)
    plt.axvline(elbow, 0, 1, color="r")
    plt.text(elbow + 0.1, 0.5, f"knee is {elbow}")

    if save_path:
        fig.savefig(save_path + f"kmeans elbow {name}.png", bbox_inches="tight")
    else:
        plt.show()
