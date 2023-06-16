import argparse
import warnings
import molli.ncsa_workflow.clustering.visualization as viz
import molli.ncsa_workflow.clustering.helpers as helpers

parser = argparse.ArgumentParser(
    'molli plot',
    description='Read the JSON file output of cluster.py and visuzalize'
)

parser.add_argument(
    'scores',
    action='store',
    type=str,
    metavar='<fpath>',
    help='Path to JSON file of dimensionality reduction scores and cluster assignments'
)

parser.add_argument(
    'exemplars',
    action='store',
    type=str,
    metavar='<fpath>',
    help=('Path to JSON file of list of list of exemplars,'
    ' s.t. index 1 coresponds to list of exemplars with k=2 clusters')
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

parser.add_argument(
    '-m',
    '--method',
    action='store',
    type=str,
    default='tsne',
    help=('The type of dimensionality reduction that was performed.'
          ' Accepted options are tsne and pca, currently.')
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

    input_scores = parsed.scores
    input_exemp = parsed.exemplars
    clusters = parsed.clusters
    output = parsed.output
    method = parsed.method
    color = parsed.color_scheme

    try:
        scores, exemp = helpers.unpack_json(input_scores, input_exemp)
    except Exception as exp:
        warnings.warn(f"Error with JSON input files: {exp!s}")

    if clusters is None:
        clusters = exemp[-1]
    else:
        try:
            temp = int(clusters)
        except ValueError: # not an int
            match clusters:
                case 'elbow':
                    clusters = exemp[-1]
                case _:
                    raise ValueError(f"Invalid cluster value: {method}")
        else:   # is an str representation of an int. Must check within bounds
            clusters = temp
            if clusters < 1 or clusters > (len(scores) - 2):    # specific to formatting of JSON file
                raise ValueError(f"Invalid cluster value: {method}")

    match method:
        case 'tsne':
            viz.tsne_plot(scores, exemp[clusters - 1], 2, 'test', output, clusters)
        case 'pca':
            viz.pca_plot(scores, exemp[clusters - 1], 2, 'test', output, clusters)
        case _:
            raise ValueError(f"Unsupported mode: {method}")