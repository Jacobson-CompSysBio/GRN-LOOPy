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
	if(series.sum() < 0.00000001):
		return_series=series.copy()
		return_series[:] = False
		return return_series
	zscore = stats.zscore(series)
	z = np.abs(zscore)
	return_series = pd.Series(z > n_stds)
	return_series.name = series.name
	return return_series

def create_outlier_sample_rows(raw_df: pd.DataFrame, n_stds: int) -> pd.DataFrame: 
	"""
	Create outliers row series
	"""

	outlier_list = []
	for column in raw_df.columns:
		outlier_series = extract_outlier_samples(raw_df[column], n_stds)

		outlier_list.append(outlier_series)
	print(type(outlier_list))
	print(len(outlier_list))
	outlier_df = pd.concat(outlier_list, axis=1)
	outlier_df.columns = raw_df.columns
	return_series = outlier_df.apply(lambda x: any(x), axis=1)

	return return_series, outlier_df 


def extract_outlier_indices_and_cols(outlier_df): 
	outlier_sum_index_mask = outlier_df.apply(sum, axis=1) >= 1
	outlier_sum_col_mask = outlier_df.sum() >= 1
	
	outlier_sum_indices = outlier_df.index[ outlier_sum_index_mask ]
	outlier_sum_cols = outlier_df.columns[ outlier_sum_col_mask ] 
	
	return outlier_sum_indices, outlier_sum_cols

def remove_high_pct_outlier_rows(raw_df, outlier_df, outlier_columns, outlier_indices,  pct_threshold=0.25): 
	"""
	this function removes the samples wherein outliers exist in a high percentage of the total number of columns 
	found to have outliers. 
	"""
	# get percentage of outliers

	total_number_of_outlier_features = len(outlier_columns)
	percentages = outlier_df[ outlier_columns ].loc[outlier_indices].sum(axis=1) / total_number_of_outlier_features
	
	# remove high percentage outliers as they're wonky across the board.
	pct_mask = percentages >= pct_threshold
	high_outlier_indices = percentages[pct_mask].index
		
	df_dropped_high_pct_outlier = raw_df.drop(high_outlier_indices)
	outlier_df_dropped_high_pct = outlier_df.drop(high_outlier_indices)

	return df_dropped_high_pct_outlier, outlier_df_dropped_high_pct


def drop_rows_with_extreme_outliers(raw_df, n_sds, nth_percentile_for_drop):
	"""
	This function calculates outliers by a zscore greater than n sds, and then 
	removes outliers above or below the nth percentile 
	""" 
	rows_with_outliers, outlier_df = create_outlier_sample_rows(raw_df, n_sds)
	outlier_indices, outlier_columns = extract_outlier_indices_and_cols(outlier_df)

	df_dropped_high_pct_outlier, outlier_df = remove_high_pct_outlier_rows(raw_df, outlier_df, np.array(outlier_columns.values), np.array(outlier_indices.values), nth_percentile_for_drop)

	return df_dropped_high_pct_outlier, outlier_df


def winsorize_data(input_file, n_sds=6, limits=0.05, has_index_col: bool=False, sep: str = '\t', verbose: bool = False) -> str: 
	"""
	This function extracts the total number of outliers, finds the columns, and then winsorizes the column
	ultimately squashing the data to the nth percentil. 
	"""

	index_col = 0 if has_index_col else None
	raw_data = pd.read_csv(input_file, sep=sep, index_col= index_col) 
	print("Raw Data: ", raw_data.head())
	rows_with_outliers, outlier_df = create_outlier_sample_rows(raw_data, n_sds)
	outlier_columns = outlier_df.columns
	print("outlier", outlier_df.shape)
	for col in outlier_columns: 
		raw_data[col] = stats.mstats.winsorize(raw_data[col], limits=0.05)
	path, base_name = os.path.split(input_file)
	output_file_name = f"{path}/outlier_removed_{base_name}" 
	print("Saving to file: ", output_file_name)
	raw_data.to_csv(output_file_name, sep=sep, index = has_index_col)
	return output_file_name




# TODO: Consider whether this normalization is even worthwhile
# def outlier_removal_and_winsorization(raw_df: pd.DataFrame, n_sds: int, pct_threshold: float =0.25, winsorize_all_data: bool = False):
#	 """
#	 This function removes outliers above or below the nth percentile
#	 """
#	 rows_with_outliers, outlier_df = create_outlier_sample_rows(raw_df, n_sds)
#	 outlier_indices, outlier_columns = extract_outlier_indices_and_cols(outlier_df)


#	 if winsorize_all_data: 
#		 """
#		 IF all data are to be winsorized, then we immediately winsorize them and return the base data
#		 """
#		 for col in outlier_columns: 
#			 raw_df[col] = stats.mstats.winsorize(raw_df[col], limits=0.05)
	
#		 return raw_df

#	 df_dropped_high_pct_outlier, outlier_df = remove_high_pct_outliers(raw_df, outlier_df, outlier_columns, outlier_indices, pct_threshold)

#	 rows_with_outliers, outlier_df = create_outlier_sample_rows(df_dropped_high_pct_outlier, n_sds)
#	 outlier_indices, outlier_columns = extract_outlier_indices_and_cols(outlier_df)


#	 return df_dropped_high_pct_outlier , outlier_columns
