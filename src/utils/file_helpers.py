import pandas as pd


def read_dataframe(filepath, sep='\t', header=True):
	df = pd.read_csv(filepath, sep=sep, header=header)

	return df