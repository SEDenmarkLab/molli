import os
os.environ['OMP_NUM_THREADS'] = '8'

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import numpy as np
import h5py
from tqdm import tqdm

# the output from molli aso calculation

# use these lines for the full in silico library (without focused analogues)
# in_aso = '../descriptors/no_linker_filter/massive_aso.h5'
# out_folder = 'postprocessed_aso/'

# use these lines for the full in silico library vs. full + the focused analogues
# in_aso = '../descriptors/full_lib_with_commercial/massive_aso.h5'
in_aso = '../descriptors/full_lib_with_commercial_and_fa/massive_aso.h5'

out_folder = f'postprocessed_aso_full_lib_with_commercial_and_fa/'

corr_coefs = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]


# reads in the h5 file in a nice way for pandas
def unpack_h5py(file):
    df = {}
    with h5py.File(file, 'r') as f:
        # print(f.keys())
        # exit()
        for x in tqdm(f.keys()):
            df[x] =np.array(f[x])
    return pd.DataFrame.from_dict(df).transpose()

# this function calculates the total variance of the ASO vector space, defined as trace of covariance sample matrix.
def calc_total_variance(data):
    covariance = data.cov()
    total = np.trace(covariance)
    return total

# removes features with variance below the threshold (default is 0)
def variance_feature_selection(data: pd.DataFrame, thld: float = 0) -> pd.DataFrame:
    threshold = VarianceThreshold(thld)
    threshold.fit(data)

    kept_features = threshold.get_support(indices = True)
    data_vt = data.iloc[:, kept_features]

    print(f'shape of data after variance threshold: {data_vt.shape}')                           # keep in?
    print(f'total variance after variance threshold: {calc_total_variance(data_vt):.2f}\n')

    return data_vt

# if no correlation coeffecient threshold is given, outputs list of tuples containing shape and variance of data after removing columns for
# [0.95, 0.90, 0.80, 0.70, 0.60, 0.50] correlation thresholds. If threshold is given then returns dataframe after column dropping
def correlated_columns(data: pd.DataFrame, coef : float = None) -> list | pd.DataFrame:
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    coeffecients = corr_coefs
    r_df = False
    if coef is not None:
        coeffecients = [coef]
        r_df = True

    results = []
    for corr_coef in coeffecients:
        to_drop = [column for column in upper.columns if any(upper[column] > corr_coef)]
        data_nocorr = data.drop(to_drop, axis=1, inplace = False)
        print(f'shape of data after removing correlated columns (R > {corr_coef}): {data_nocorr.shape}')
        print(f'total variance after removing correlated columns (R > {corr_coef}): {calc_total_variance(data_nocorr):.2f}\n')
        results.append((data_nocorr.shape, calc_total_variance(data_nocorr)))

    if r_df:
        return data_nocorr
    else:
        return results

if __name__ == '__main__':

    df = unpack_h5py('./massive_aso_nolink.h5')
    print(df)

    print(df.shape)
    print(calc_total_variance(df))

    dfv = variance_feature_selection(df)
    temp = correlated_columns(dfv)
    print(temp)
    dfvc = correlated_columns(dfv, 0.8)
    print(dfvc)

    with open('../../../out_conformers1/nolink_post_processed_df' + '.csv', 'w') as o:
            dfvc.to_csv(o)


'''
    data=unpack_h5py(in_aso)
    print(data)
    print(f'initial data shape: {data.shape}')
    # exit()
    print(f'initial total variance: {calc_total_variance(data):.2f}\n')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # remove zero variance columns, save
    threshold = VarianceThreshold()
    threshold.fit(data)
    # this allows us to keep the indices of the gridpoints in the raw aso output
    kept_features = threshold.get_support(indices = True)
    data_vt = data.iloc[:, kept_features]
    print(f'shape of data after variance threshold: {data_vt.shape}')
    print(f'total variance after variance threshold: {calc_total_variance(data_vt):.2f}\n')
    with open(out_folder + 'aso_vt0.csv', 'w') as o:
        data_vt.to_csv(o)

    # get correlation coefficients
    corr_matrix = data_vt.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    for corr_coef in corr_coefs:
        to_drop = [column for column in upper.columns if any(upper[column] > corr_coef)]
        data_vt_nocorr = data_vt.drop(to_drop, axis=1, inplace = False)
        print(f'shape of data after removing correlated columns (R > {corr_coef}): {data_vt_nocorr.shape}')
        print(f'total variance after removing correlated columns (R > {corr_coef}): {calc_total_variance(data_vt_nocorr):.2f}\n')

        # save file
        f_name = f'aso_vt0_nocorr{corr_coef * 100}'.replace('.', '')
        nocorr_dir = f'{out_folder}nocorr_extracted_aso/'
        if not os.path.exists(nocorr_dir):
            os.makedirs(nocorr_dir)
        with open(nocorr_dir + f_name + '.csv', 'w') as o:
            data_vt_nocorr.to_csv(o)

        # doo 200D PCA projection, save file
        pca_dir = f'{out_folder}pca_projections/'
        if not os.path.exists(pca_dir):
            os.makedirs(pca_dir)
        pca = PCA(n_components = 200, random_state = 42)
        data_vt_pca200 = pd.DataFrame( pca.fit_transform(data_vt_nocorr), index=data_vt_nocorr.index)
        print(f'explained variance in 200D PCA projection: {sum(pca.explained_variance_ratio_)}\n')
        with open(pca_dir + f_name + 'pca200.csv', 'w') as o:
            data_vt_pca200.to_csv(o)

    print('Success!')
'''