import pandas as pd
import numpy as np

def read_dataframe(filepath, sep='\t', header=None, has_indices=None):
	header_row = 0 if header is not None else None
	index_col = 0 if has_indices is not None else None
	df = pd.read_csv(filepath, sep=sep, header=header_row, index_col=index_col)
	return df

