import pandas as pd
import numpy as np

def read_dataframe(filepath, sep='\t', header=None):
	df = pd.read_csv(filepath, sep=sep, header=header)
	return df

