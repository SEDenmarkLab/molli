import os


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

# use these lines for the full in silico library plus the focused analogues
in_aso = 'test_aso/massive_aso.h5'

out_folder = 'postprocessed_aso_full_lib_with_commercial/'

corr_coefs = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]


# reads in the h5 file in a nice way for pandas
def unpack_h5py(file):
    df = {}
    with h5py.File(file, 'r') as f:
        for x in tqdm(f.keys()):
            df[x] =np.array(f[x])
    return pd.DataFrame.from_dict(df).transpose()

# this function calculates the total variance of the ASO vector space, defined as trace of covariance sample matrix.
def calc_total_variance(data):
    covariance = data.cov()
    total = np.trace(covariance)
    return total


if __name__ == '__main__':

    data=unpack_h5py(in_aso)
    print(data)
    exit()
    print(f'initial data shape: {data.shape}')
    print(f'initial total variance: {calc_total_variance(data):.2f}\n')

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
        f_out = f'aso_vt0_nocorr{corr_coef * 100}'
        with open(out_folder + f_out + '.csv', 'w') as o:
            data_vt_nocorr.to_csv(o)

        # doo 200D PCA projection, save file
        pca = PCA(n_components = 200, random_state = 42)
        data_vt_pca200 = pd.DataFrame( pca.fit_transform(data_vt_nocorr), index=data_vt_nocorr.index)
        print(f'explained variance in 200D PCA projection: {sum(pca.explained_variance_ratio_)}\n')
        with open(out_folder + f_out + 'pca200.csv', 'w') as o:
            data_vt_pca200.to_csv(o)

    print('Success!')