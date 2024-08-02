import os
import pandas as pd
from pathlib import Path
import sys
import warnings

def extract_edgelist(df, rep): 
	if df is None: 
		return pd.DataFrame()
	edgelist_colnames = df.columns
	to_mask = df[edgelist_colnames[0]] == rep
	from_mask = df[edgelist_colnames[1]] == rep
	rep_edgelist = df[ to_mask | from_mask ]
	return rep_edgelist

def get_represented_genes_from_correlation_df(correlate_df, rep): 
	non_representatives_in_map = correlate_df[correlate_df['representative'] == rep]
	unique_non_reps = non_representatives_in_map['non_representative'].unique()
	return unique_non_reps

def copy_non_rep_to_representative_df(network_edgelist_df: pd.DataFrame, correlated_data_df: pd.DataFrame) -> pd.DataFrame:
	non_rep_edgelist_list = []
	# Get the unique representatives
	representative_list = correlated_data_df['representative'].unique()
	# edgelist_colnames = network_edgelist_df.columns
	output_df = None
	for rep in representative_list:
		rep_edgelist = extract_edgelist(network_edgelist_df, rep)
		reps_in_output_df = extract_edgelist(output_df, rep)
		unique_non_reps = get_represented_genes_from_correlation_df(correlated_data_df, rep)
		if rep_edgelist.shape[0] == 0 and reps_in_output_df.shape[0] == 0 :
			continue
		for unique_nonrep in unique_non_reps:
			if unique_nonrep == rep: 
				continue
			unique_non_rep_edgelist = rep_edgelist.copy().replace(rep, unique_nonrep)
			unique_non_rep_edgelist_from_output_df = reps_in_output_df.copy().replace(rep, unique_nonrep)
			output_df = pd.concat([output_df, unique_non_rep_edgelist, unique_non_rep_edgelist_from_output_df])
	# if len(non_rep_edgelist_list) == 0: 
	# 	base_warn = "No representatives from supplied file exist in network.\n"
	# 	suggestion = "This could be caused by using the wrong representative map file\n"
	# 	suggestion2 = " or setting a threshold too high. "
	# 	warnings.warn(f"{base_warn} {suggestion} {suggestion2}")
	# 	return None
	# output_df = pd.concat(non_rep_edgelist_list).reset_index(drop=True)
	return output_df.sort_values(by='weight', ascending=False).reset_index(drop=True)


def add_correlates_back_to_df(network_edgelist_df: pd.DataFrame, correlated_data_file_path: str):
	# network_edgelist_df = pd.read_csv(network_file_path, sep='\t', header=None, index_col=None)
	correlated_data_df = pd.read_csv(correlated_data_file_path, sep='\t')
	if correlated_data_df.shape[0] == 0: 
		return(network_edgelist_df)
	# # Get the unique representatives
	# representative_list = correlated_data_df['representative'].unique()
	# get the correlates and then concat them to original network
	print("Correlated_data df") 
	print(correlated_data_df)
	non_rep_concatenation = copy_non_rep_to_representative_df(network_edgelist_df, correlated_data_df)#, representative_list)
	total_network = pd.concat([network_edgelist_df, non_rep_concatenation]).reset_index(drop=True)
	print("TOTAL NETWORK") 
	print(total_network) 
	# get new filename.
	return total_network

