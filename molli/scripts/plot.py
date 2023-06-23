import argparse
import warnings
import molli.ncsa_workflow.clustering.visualization as viz
import molli.ncsa_workflow.clustering.helpers as helpers

parser = argparse.ArgumentParser(
    'molli plot',
    description='Read the JSON file output of cluster.py and visuzalize'
)

parser.add_argument(
    'input',
    action='store',
    type=str,
    metavar='<fpath>',
    help='Path to JSON file of dimensionality reduction output'
)

parser.add_argument(
    '-k',
    '--clusters',
    action='store',
    default='elbow',
    help=(
        'Number of clusters to represent in plot. Must be between 1'
        ' and max k of JSON file, inclusive. Defaults to "elbow"'
        ' of k-means cluster plot'
        )
)

parser.add_argument(
    '-o',
    '--output',
    action='store',
    type=str,
    metavar='<fpath>',
    help='Output filepath of plot as a png file'
)

# TO-DO
parser.add_argument(
    '-c',
    '--color_scheme',
    type=str,
    default=None,
    help='TO BE IMPLEMENTED: Select from list of available color schemes'
)

def molli_main(args, config=None, output=None, **kwargs):
    parsed = parser.parse_args(args)

    input = parsed.input
    clusters = parsed.clusters
    output = parsed.output
    color = parsed.color_scheme

    try:
        scores, assignments, exemplars, knee, method, _ = helpers.unpack_json(input)
    except Exception as exp:
        warnings.warn(f"Error with JSON input files: {exp!s}")

    if clusters is None:
        clusters = knee
    else:
        try:
            temp = int(clusters)
        except ValueError: # not an int
            match clusters:
                case 'elbow':
                    clusters = knee
                case _:
                    raise ValueError(f"Invalid cluster value: {clusters}")
        else:   # is an str representation of an int. Must check within bounds
            clusters = temp
            if clusters < 1 or clusters > (len(assignments)):    # specific to formatting of JSON file
                raise ValueError(f"Invalid cluster value: {clusters}")
    print(knee)
    match method:
        case 'tsne':
            viz.tsne_plot(scores, assignments[str(clusters)], exemplars[clusters - 1], 2, 'test', output, clusters)
        case 'pca':
            viz.pca_plot(scores, assignments[str(clusters)], exemplars[clusters - 1], 2, 'test', output, clusters)
        case _:
            raise ValueError(f"Unsupported mode: {method}")