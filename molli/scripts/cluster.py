'''
Run projection and kmeans cluster analysis on descriptor data
'''

import argparse
import warnings

import molli.ncsa_workflow.clustering.helpers as help
import molli.ncsa_workflow.clustering.post_processing as pp
import molli.ncsa_workflow.clustering.dimension_reduction as dr    # check this out later

parser = argparse.ArgumentParser(
    'molli cluster',
    description='Runs tsne or PCA and kmeans clustering on aso descriptor data'
)

parser.add_argument(
    "input",
    action='store',
    metavar='<fpath>',
    help='ASO description of conformer library as a h5 file'
)

parser.add_argument(
    '--output',
    '-o',
    action='store',
    required=True,
    metavar='<fpath>',
    help=(
        'filepath for directory of output of analysis as two json files. First is a DataFrame with indices as catalyst ids'
        ' and 2d analysis values and cluster assignment by number of clusters as columns. Second is a'
        ' list of lists of exemplars, list index corresponding to k-1 number of clusters'
    )
)

parser.add_argument(
    '--variance',
    '-v',
    action='store',
    default=0,
    help=(
        'Optional parameter for variance based feature selection. Removes features at or below given threshold,'
        ' default value of 0. Set to "false" to disable'
    )
)

parser.add_argument(
    '--correlation',
    '-c',
    action='store',
    default=0.8,
    help=(
        'Optional parameter for dropping highly correlated columns. Removes columns above given threshold, given as a float [0, 1]'
        ', default value of 0.8. Set to "false" to disable'
    )
)

parser.add_argument(
    '--mode',
    '-m',
    action='store',
    default='tsne',
    type=str,
    help=(
        'Method of projection/dimensionality-reduction to be performed on the data.'
        ' Currently accepted inputs are tsne or pca. Defaults to tsne'
    )
)

parser.add_argument(
    '--perplexity',
    '-p',
    action='store',
    default=5,
    type=int,
    help=(
        'Perplexity for tSNE projection. Defaults to 5.'
    )
)

parser.add_argument(
    '--k_clusters',
    '-k',
    action='store',
    default=20,
    type=int,
    help='Integer for inclusive upper bound of number of clusters for k-means cluster analysis. Defaults to 20.'
)


def molli_main(args, config=None, output=None, **kwargs):
    parsed = parser.parse_args(args)

    try:
        df = help.unpack_h5py(parsed.input)
    except Exception as exp:
        warnings.warn(f'Issue with descriptor file input: {exp!s}')
        
    try:
        if parsed.variance != 'false':
            if float(parsed.variance) >= 0 and float(parsed.variance) <= 1:
                df = pp.variance_feature_selection(df, float(parsed.variance))
            else:
                raise ValueError('Invalid variance value')
        if parsed.correlation != 'false':
            if float(parsed.correlation) >= 0 and float(parsed.correlation) <= 1:
                df = pp.correlated_columns(df, float(parsed.correlation))
            else:
                raise ValueError('Invalid correlation value')
    except Exception as exp:
        warnings.warn(f'Issue with variance or correlation feature selection: {exp!s}')

    try:
        assert parsed.k_clusters > 0
    except Exception as exp:
        warnings.warn(f'Invalid k_clusters argument, must be greater than 0: {exp!s}')

    match parsed.mode:  # where to check output filepath validity? in data_processing.processed_json right now
        case 'tsne':
            dr.tsne_processing(df, parsed.output, parsed.perplexity, parsed.k_clusters)
        case 'pca':
            dr.pca_processing(df, parsed.output, parsed.k_clusters)
        case _:
            warnings.warn(f'Unknown mode: {parsed.mode}')

    # issue with intel openmp and llvm openmp???