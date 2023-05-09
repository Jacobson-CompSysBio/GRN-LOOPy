"""
This Module contains helpers for normalizing data and removing
the low variance features.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats



def normalize_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function normalizes data from 0 to 1
    """
    raw_df = raw_df / raw_df.sum()

    raw_df = raw_df.fillna(0)
    return raw_df


def extract_outlier_samples(series: pd.Series, n_stds: int) -> pd.Series: 
    """
    This function returns true if the values within excede n standard deviations
    from the mean
    """
    zscore = stats.zscore(series)
    z = np.abs(zscore)

    return z > n_stds

def create_outlier_sample_rows(raw_df: pd.DataFrame, n_stds: int) -> pd.DataFrame: 
    """
    Create outliers row series
    """

    outlier_df = pd.DataFrame()
    for column in raw_df.columns:
        outlier_series = extract_outlier_samples(raw_df[column])

        outlier_df[column] = outlier_series

    outlier_series = outlier_df.apply(lambda x: any(x), axis=1)

    return outlier_series 


def remove_outliers(raw_df, n_sds):
    """
    This function removes outliers above or below the nth percentile
    """
    print("hi")
    


def remove_low_variance_features(raw_df: pd.DataFrame, variance_thresh: float, print_meta: bool = False, path:str ='./') -> pd.DataFrame:
    """
    This function removes features with variance that
    falls under the threshold variance_thresh
    """
    
    threshold_mask = raw_df.var() >= variance_thresh
    high_var_columns = raw_df.columns[threshold_mask]

    low_variance_dataframe = raw_df[ high_var_columns ]

    if len(low_variance_dataframe.columns) == 0: 
        raise RuntimeError("remove_low_variance removed all features from dataframe.")

    if print_meta:
        low_var_cols = raw_df.columns[ ~threshold_mask ]
        with open(f'{path}/meta.txt', 'a') as f:
            f.write('num_low_var_removed_cols\t', len(low_var_cols))
            f.write("low_var_removed_cols\t", low_var_cols)

    return low_variance_dataframe


def write_high_var_to_file(high_var_df: pd.DataFrame, variance_thresh: float, filename: str, has_index_col: bool) -> str:
    """
    This function writes the high var dataframe to file and then
    returns the new file name.
    """
    base_name = '.'.join(filename.split('.')[0:-1])
    
    outfile_name = f"{base_name}_ge{variance_thresh}variance.tsv"
    
    high_var_df.to_csv(outfile_name, sep='\t', index=has_index_col)
    
    return outfile_name

def remove_low_var_and_save(input_file: str, variance_thresh: float, has_index_col: bool, sep: str = '\t', print_meta: bool = True) -> str: 
    """
    This function combines the two above functions and reorders the dataframe
    to ensure that the original index columns are not lost.
    """

    index_col = 0 if has_index_col else None
    raw_data = pd.read_csv(input_file, sep=sep, index_col= index_col)
    
    high_var_df = remove_low_variance_features(raw_data, variance_thresh, print_meta, os.path.dirname(input_file))

    outfile_name = write_high_var_to_file(high_var_df, variance_thresh, input_file, has_index_col)

    return outfile_name
