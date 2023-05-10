"""
This Module contains helpers for finding and removing
outlier samples.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

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

    outlier_list = []
    for column in raw_df.columns:
        outlier_series = extract_outlier_samples(raw_df[column], n_stds)

        outlier_list.append(outlier_series)

    outlier_df = pd.concat(outlier_list, axis=1)
    outlier_df.columns = raw_df.columns
    return_series = outlier_df.apply(lambda x: any(x), axis=1)

    return return_series, outlier_df 


# def winsorize_data(outlier_df: pd.DataFrame): 
#     outlier_row_sums_mask = outlier_df.apply(sum, axis=1) >= 1
#     outlier_indices = outlier_df.loc[ outlier_row_sums_mask ].index
    
#     outlier_col_sum_mask = outlier_df_6.sum() >= 1
#     outlier_cols = outlier_df.columns[ outlier_col_sum_mask ]
    
def extract_outlier_indices_and_cols(outlier_df): 
    outlier_sum_index_mask = outlier_df.apply(sum, axis=1) >= 1
    outlier_sum_col_mask = outlier_df.sum() >= 1
    
    outlier_sum_indices = outlier_df.index[ outlier_sum_index_mask ]
    outlier_sum_cols = outlier_df.columns[ outlier_sum_col_mask ] 
    
    return outlier_sum_indices, outlier_sum_cols

def remove_high_pct_outliers(raw_df, outlier_df, outlier_columns, outlier_indices, total_number_of_outlier_features, pct_threshold=0.25): 
    """
    this function removes the samples wherein outliers exist in a high percentage of the total number of columns 
    found to have outliers. 
    """
    # get percentage of outliers
    percentages = outlier_df[ outlier_columns ].loc[outlier_indices].sum(axis=1) / total_number_of_outlier_features
    
    # remove high percentage outliers as they're wonky across the board.
    pct_mask = percentages >= pct_threshold
    high_outlier_indices = percentages[pct_mask].index
        
    print(percentages[pct_mask])
        
    df_dropped_high_pct_outlier = raw_df.drop(high_outlier_indices)
    outlier_df_dropped_high_pct = outlier_df.drop(high_outlier_indices)
    
    return df_dropped_high_pct_outlier, outlier_df_dropped_high_pct

def outlier_removal(raw_df: pd.DataFrame, n_sds: int, pct_threshold: float =0.25, winsorize_all_data: bool = False):
    """
    This function removes outliers above or below the nth percentile
    """
    rows_with_outliers, outlier_df = create_outlier_sample_rows(raw_df, n_sds)
   
    outlier_indices, outlier_columns = extract_outlier_indices_and_cols(outlier_df)
    
    if winsorize_all_data: 
        for col in outlier_columns: 
            raw_df[col] = winsorize(raw_df[col], limits=0.05)
    
        return raw_df

    # get the total columns in which there exist outliers
    total_number_of_outlier_features = len(outlier_columns)

    df_dropped_high_pct_outlier, outlier_df = remove_high_pct_outliers(raw_df, outlier_df, outlier_columns, outlier_indices, total_number_of_outlier_features, pct_threshold)

    rows_with_outliers, outlier_df = create_outlier_sample_rows(df_dropped_high_pct_outlier, n_sds)
    outlier_indices, outlier_columns = extract_outlier_indices_and_cols(outlier_df)
    
    # for col in outlier_columns: 
    #     print(col)
    #     plt.figure(figsize=(12,3))
    #     plt.subplot(1, 2, 1)
    #     plt.hist(df_dropped_high_pct_outlier[col])
    #     plt.title(col)
    #     df_dropped_high_pct_outlier[col] = winsorize(df_dropped_high_pct_outlier[col], limits=(0.05, 0.05))
    #     plt.subplot(1, 2, 2)
    #     plt.hist(df_dropped_high_pct_outlier[col])
    #     plt.title(col + 'winsor')
    return df_dropped_high_pct_outlier , outlier_columns
    