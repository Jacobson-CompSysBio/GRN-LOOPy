import pandas as pd

"""
This file contains functions for normalization of raw data input files.
"""

# def write_df_to_file(df: pd.DataFrame, variance_thresh: float, filename: str, has_index_col: bool) -> str:
# 	"""
# 	This function writes the high var dataframe to file and then
# 	returns the new file name.
# 	"""
	
	
# 	outfile_name = f"{base_name}_ge{variance_thresh}variance.tsv"
	
	
	
# 	return outfile_name

def normalize_data_set(input_file: str,  has_index_col: bool, sep: str = '\t', verbose: bool = False) -> str: 
	"""
	This function combines the two above functions and reorders the dataframe
	to ensure that the original index columns are not lost.
	"""

	index_col = 0 if has_index_col else None
	raw_data = pd.read_csv(input_file, sep=sep, index_col= index_col)
	scalar = 1000 
	# L1 normalize data: 
	raw_data = scalar * raw_data / raw_data.abs().sum()
	
	base_name = '.'.join(input_file.split('.')[0:-1])
	outfile_name = f"{base_name}_l1_normalized.tsv"
	print(raw_data)
	raw_data.to_csv(outfile_name, sep='\t', index=has_index_col)

	return outfile_name
