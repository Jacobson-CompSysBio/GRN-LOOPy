import os
import pandas as pd
from pathlib import Path
import sys
import warnings

def create_non_rep_df(network_edgelist_df: pd.DataFrame, correlated_data_df: pd.DataFrame) -> pd.DataFrame:
	non_rep_edgelist_list = []

	# Get the unique representatives
	representative_list = correlated_data_df['representative'].unique()
	edgelist_colnames = network_edgelist_df.columns

	for rep in representative_list:
		to_mask = network_edgelist_df[edgelist_colnames[0]] == rep
		from_mask = network_edgelist_df[edgelist_colnames[1]] == rep
		rep_edgelist = network_edgelist_df[ to_mask | from_mask ]
		non_representatives_in_map = correlated_data_df[correlated_data_df['representative'] == rep]
		unique_non_reps = non_representatives_in_map['non_representative'].unique()

		if len(rep_edgelist) == 0:
			continue

		for unique_nonrep in unique_non_reps:
			if unique_nonrep == rep: 
				continue
			unique_non_rep_edgelist = rep_edgelist.copy().replace(rep, unique_nonrep)
			non_rep_edgelist_list.append(unique_non_rep_edgelist)

	if len(non_rep_edgelist_list) == 0: 
		base_warn = "No representatives from supplied file exist in network.\n"
		suggestion = "This could be caused by using the wrong representative map file\n"
		suggestion2 = " or setting a threshold too high. "
		warnings.warn(f"{base_warn} {suggestion} {suggestion2}")
		return None

	output_df = pd.concat(non_rep_edgelist_list).reset_index(drop=True)
	return output_df.sort_values(by='weight', ascending=False).reset_index(drop=True)


def add_correlates_back_to_df(network_edgelist_df: pd.DataFrame, correlated_data_file_path: str):
	# network_edgelist_df = pd.read_csv(network_file_path, sep='\t', header=None, index_col=None)
	correlated_data_df = pd.read_csv(correlated_data_file_path, sep='\t')

	# # Get the unique representatives
	# representative_list = correlated_data_df['representative'].unique()

	# get the correlates and then concat them to original network
	non_rep_concatenation = create_non_rep_df(network_edgelist_df, correlated_data_df)#, representative_list)
	total_network = pd.concat([network_edgelist_df, non_rep_concatenation]).reset_index(drop=True)

	# get new filename.
	return total_network

