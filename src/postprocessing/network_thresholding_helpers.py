import itertools
import numpy as np
import os
import pandas as pd
import time
import sys

def threshold_edgelist(sorted_data: pd.DataFrame, top_pct: float) -> pd.DataFrame:
	"""
	This function takes in an edgelist and slices the top n'th percent 
	defined by the `top_pct` and returns a dataframe. 
	"""
	sorted_thresh_pct_idx = int( len(sorted_data) * top_pct ) -1 
	sorted_threshold_df = None
	sorted_threshold_df = sorted_data.loc[:sorted_thresh_pct_idx]
	return sorted_threshold_df

def sort_edgelist(df: pd.DataFrame, sort_column:str = '') -> pd.DataFrame: 
	""""
	This function takes in a dataframe, sorts it based upon the "weight" 
	column (i.e. the third column of three).
	"""
	sort_column = sort_column if sort_column != '' else df.columns[2]
	df = df.sort_values(by=[sort_column], ascending=False).reset_index(drop=True)
	return df
