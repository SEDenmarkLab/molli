import os
os.environ['OMP_NUM_THREADS'] = '8'

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
# the output from molli aso calculation

# use these lines for the full in silico library (without focused analogues)
# in_aso = '../descriptors/no_linker_filter/massive_aso.h5'
# out_folder = 'postprocessed_aso/'

# use these lines for the full in silico library vs. full + the focused analogues
# in_aso = '../descriptors/full_lib_with_commercial/massive_aso.h5'
in_aso = '../descriptors/full_lib_with_commercial_and_fa/massive_aso.h5'

out_folder = f'postprocessed_aso_full_lib_with_commercial_and_fa/'

corr_coefs = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]


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
# remove this functionality? Only dataframe from given correlation threshold
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