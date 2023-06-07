'''
Run projection and kmeans cluster analysis on descriptor data
'''

import argparse
import warnings

import molli as ml
import molli.lib_gen.test_aso.post_processing as pp
import molli.lib_gen.test_aso.data_processing as dp    # check this out later


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

parser.add_argument(
    '--plot',
    action='store_true',
    help='Boolean to display matplotlib visualization of your analysis. Defaults to False'
)


def molli_main(args, config=None, output=None, **kwargs):   #TODO
    parsed = parser.parse_args(args)

    try:
        df = pp.unpack_h5py(parsed.input)
    except Exception as exp:
        warnings.warn(f'Issue with descriptor file input: {exp!s}') # how should warning work?

    try:
        if parsed.variance != 'false':
            if parsed.variance >= 0 and parsed.variance <= 1:
                df = pp.variance_feature_selection(df, parsed.variance)
            else:
                raise ValueError('Invalid variance value')
        if parsed.correlation != 'false':
            if parsed.correlation >= 0 and parsed.correlation <= 1:
                df = pp.correlated_columns(df, parsed.correlation)
            else:
                raise ValueError('Invalid correlation value')
    except Exception as exp:
        warnings.warn(f'Issue with variance or correlation feature selection: {exp!s}')

    match parsed.mode:  # where to check output filepath validity?
        case 'tsne':
            dp.tsne_processing(df, parsed.output, parsed.perplexity, parsed.k_clusters, parsed.plot)
        case 'pca':
            dp.pca_processing(df, parsed.output, parsed.k_clusters, parsed.plot)
        case _:
            warnings.warn(f'Unknown mode: {parsed.mode}')
