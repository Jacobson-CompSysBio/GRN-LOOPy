import pandas as pd
import numpy as np

def read_dataframe(filepath, sep='\t', header=None, index_col=None):
	df = pd.read_csv(filepath, sep=sep, header=header, index_col=index_col)
	return df

