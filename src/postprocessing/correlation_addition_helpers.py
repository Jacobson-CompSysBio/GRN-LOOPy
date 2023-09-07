import os
import pandas as pd
from pathlib import Path
import sys


def create_non_rep_df(network_edgelist_df: pd.DataFrame, correlated_data_df: pd.DataFrame, representative_list: list) -> pd.DataFrame:
	non_rep_edgelist_list = []

	for rep in representative_list:
		to_mask = network_edgelist_df[0] == rep
		from_mask = network_edgelist_df[1] == rep
		rep_edgelist = network_edgelist_df[ to_mask | from_mask ]
		non_representatives_in_map = correlated_data_df[correlated_data_df['representative'] == rep]
		unique_non_reps = non_representatives_in_map['non_representative'].unique()

		if len(rep_edgelist) == 0:
			continue

		for unique_nonrep in unique_non_reps:
			unique_non_rep_edgelist = rep_edgelist.copy().replace(rep, unique_nonrep)

			non_rep_edgelist_list.append(unique_non_rep_edgelist)

	return pd.concat(non_rep_edgelist_list).reset_index(drop=True)


def add_correlates_back_to_df(network_edgelist_df: pd.DataFrame, correlated_data_file_path: str):
	# network_edgelist_df = pd.read_csv(network_file_path, sep='\t', header=None, index_col=None)
	correlated_data_df = pd.read_csv(correlatd_data_file_path, sep='\t')

	# Get the unique representatives
	representative_list = correlated_data_df['representative'].unique()

	# get the correlates and then concat them to original network
	non_rep_concatenation = create_non_rep_df(network_edgelist_df, correlated_data_df, representative_list)
	total_network = pd.concat([network_edgelist_df, non_rep_concatenation])

	# get new filename.
	return total_network

