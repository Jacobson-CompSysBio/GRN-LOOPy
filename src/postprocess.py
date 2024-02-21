from utils.file_helpers import read_dataframe
from postprocessing.create_edgelist import create_edgelist
from postprocessing.network_thresholding_helpers import threshold_edgelist, sort_edgelist
from postprocessing.correlate_addition_helpers import add_correlates_back_to_df
from postprocessing.transform_edgelist import transform_edgelist_to_undirected
import argparse 


def get_arguments():
	"""
	Extracts command line arguments. 
	"""
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--infile', type=str, dest='infile', required=True,
						help='the base input file dataframe')
	parser.add_argument('--delim', dest='delim', action='store', default="\t",
					   help='specifies the delimiter of the file')
	parser.add_argument('--threshold', dest='threshold', action='store', default=0.05, type=float,
						help='The top nth percent of the desired network. Default 0.05')
	parser.add_argument('--rep_map_path', dest='rep_path', action='store', default=None,
						help='The file path to the representative map set.')
	parser.add_argument('--make_undirected', dest='make_undirected', action='store_true',
						help="removes low variance elements using the cv_thresh from data and saves.")
	parser.add_argument('--weigh_edges_by_acc', dest='weigh_edges_by_acc', action='store_true',
						help='weighs edges in output edgelist by accuracy score')
	parser.add_argument('--outfile', dest='outfile', action='store', default=None,
						help='the base name for the output files. Default is preprocessed.tsv')
	parser.add_argument('--verbose', dest='verbose', action='store_true',
						help='prints verbosely')

	return parser.parse_args()

def main():
	"""
	Main Function
	"""
	# get parameters
	args = get_arguments()

	infile = args.infile
# 	has_indices = args.has_indices
	delim = args.delim
	threshold = args.threshold
	rep_path = args.rep_path
	make_undirected = args.make_undirected
	weigh_by_acc = args.weigh_edges_by_acc
	outfile = args.outfile
	verbose = args.verbose

	#read output file (either edgelist or output from process)
		# create edgelist from file 
	if verbose:
		print("Reading dataframe")
	df = read_dataframe(infile, sep=delim, index_col=0, header=0)
	if verbose:
		print(df.head())
	# create network edgelist
	if verbose:
		print("Creating edgelist")
	network_edgelist = create_edgelist( df [ ~df.feature_imps.isna() ], weigh_by_acc)

	# sort the edgelist 
	if verbose: 
		print("Sorting Edgelist")
	sorted_edgelist = sort_edgelist(network_edgelist, 'weight')
	
	if make_undirected: 
		if verbose: 
			print("Collapsing edges to undirected")
			print(sorted_edgelist)
		sorted_edgelist = transform_edgelist_to_undirected(sorted_edgelist)

	print(sorted_edgelist)
	# threshold the network 
	if verbose: 
		print("Thresholding edgelist")
	thresholded_edgelist = threshold_edgelist(sorted_edgelist, threshold)

	# add in correlates
	if rep_path is not None: 
		if verbose: 
			print("Adding back correlates")
		thresholded_edgelist = add_correlates_back_to_df(thresholded_edgelist, rep_path)
	print(thresholded_edgelist)
	if outfile is None: 
		outfile = f"network_edgelist_top_{threshold}.tsv"
	thresholded_edgelist.to_csv(outfile, sep='\t')


if __name__ == "__main__": 
	main()
