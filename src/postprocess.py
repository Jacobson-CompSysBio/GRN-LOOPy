import pandas as pd
from utils.file_helpers import read_dataframe
import argparse

def get_arguments():
	"""
	Extracts command line arguments. 
	"""
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--infile', type=str, dest='infile', required=True,
						help='the base input file dataframe')
	parser.add_argument('--has_indices', dest='has_indices', action='store_true',
					   help='signifies that dataset does not have indices')
	parser.add_argument('--delim', dest='delim', action='store', default="\t"
					   help='specifies the delimiter of the file')
	parser.add_argument('--threshold', dest='threshold', action='store', default=0.05,
						help='The top nth percent of the desired network. Default 0.05')
	parser.add_argument('--rep_map_path', dest='rep_path', action='store', default=None,
						help='The file path to the representative map set.')
	parser.add_argument('--make_undirected', dest='make_undirected', action='store_true',
						help="removes low variance elements using the cv_thresh from data and saves.")
	parser.add_argument('--outfile', dest='outfile', action='store', default='preprocessed.tsv',
						help='the base name for the output files. Default is preprocessed.tsv')
	parser.add_argument('--verbose', dest='verbose', action='store_true',
						help='prints verbosely')

	return parser.parse_args()

def parse_importances(importance_data):
	features = importance_data['features'].split('|')
	feature_importances = importance_data['feature_imps'].split('|')
	feature_name = importance_data['feature'] 
	return pd.DataFrame({
		'from': features,
		'to': feature_name,
		'weight': feature_importances
	})

def create_edgelist(df: pd.DataFrame): 
	output_values = df.apply(parse_importances, axis=1)
	output_edgelist = pd.concat(output_values.values)
	return output_edgelist.sort_values(by='weight', ascending=False).reset_index(drop=True)

def threshold_edgelist(df: pd.DataFrame, threshold: float):
	threshold_index = int(threshold * df.shape[0])
	return df.loc[:threshold_index]	



def main():
	"""
	Main Function
	"""
	# get parameters
	args = get_arguments()

	infile = args.infile
	has_indices = args.has_indices
	threshold = args.threshold
	rep_path = args.rep_path
	make_undirected = args.make_undirected
	outfile = args.outfile
	verbose = args.verbose


	#read output file (either edgelist or output from process)
		# create edgelist from file 
	df = read_dataframe(infile, sep=delim, header=header_idx)
	
	total_edgelist = create_edgelist(df)

	# threshold the network 
	thresholded_edgelist = threshold_edgelist(total_edgelist, threshold)

	# add in correlates
	
	# make undirected if desired. 


if __name__ == "__main__": 
	main()
