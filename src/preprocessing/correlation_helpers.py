import numpy as np
import pandas as pd
from utils import file_helpers

### CORRELATION Functions
def correlate_data(df: pd.DataFrame, has_index_col: bool) ->  pd.DataFrame:
	"""
	Correlates data. Depending on whether an index column exists,
	the first column is removed for correlation.

	This assumes entire matrix is numerical (minus the index column).
	"""

	columns_for_correlation = df.columns[1:] if has_index_col else df.columns

	data_for_correlation = df[ columns_for_correlation ]
	reshaped = data_for_correlation.values.reshape(list(reversed(data_for_correlation.shape)))
	total_corrmat = np.corrcoef(reshaped)
	total_corrmat = pd.DataFrame(total_corrmat)
	total_corrmat.columns = columns_for_correlation
	total_corrmat.index = columns_for_correlation
	return total_corrmat

def extract_correlates_to_upper_right(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Removes the duplicated values of the correlation matrix and
	additionally removes the diagonal of the matrix.

	this assumes the diagonal has all 1s
	"""
	npdf = np.triu(df)
	colnames = df.columns
	df = pd.DataFrame(npdf, columns = colnames, index=colnames)

	# remove identity
	df = df - (df * np.eye(df.shape[1]))

	# replace na
	return df[ df.abs() >0 ]

def stack_data(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
	"""
	This stacks data into a 1 to 1 of the correlation matrix.
	"""
	df = df[ df.abs() > threshold ]
	df = df.stack(dropna=True).reset_index()
	stack_values = df.columns[-1]
	print(df)
	print('final column', stack_values)
	print(df[stack_values].abs())
	print(type(threshold))
	print(df.head())
	df.columns = ['from', 'to', 'corr']
	return df.reset_index(drop = True)

def create_correlation_list(filename: str, has_indices: bool, corr_thresh: float, save_corr: bool = False):
	"""
	This function assumes files are tsvs, correlates data, and thresholds the
	final correlation data. 

	Data are then saved to file. 
	"""
	index_col = 0 if has_indices else None
	df = file_helpers.read_dataframe(filename, header=0, sep='\t', index_col=index_col)
	print(df.head(), flush=True) 
	df = correlate_data(df, has_indices)
	print('post correlation', flush=True)
	df = extract_correlates_to_upper_right(df)
	print('post extraction', flush=True) 
	df = stack_data(df, corr_thresh)
	
	print('done', flush=True)
	base_name = '.'.join(filename.split('.')[0:-1])
	outfile_name = f"{base_name}_correlation_over_{corr_thresh}.tsv"

	if save_corr: 
		df.to_csv(outfile_name, sep='\t', index=None)
		print(f"Correlation data saved @ {outfile_name}")
	
	return df

