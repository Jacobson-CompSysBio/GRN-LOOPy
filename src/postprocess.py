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
	parser.add_argument('--threshold', dest='threshold', action='store', default='0.05', type=str,
						help='The top nth percent of the desired network. Default 0.05')
	parser.add_argument('--rep_map_path', dest='rep_path', action='store', default=None,
						help='The file path to the representative map set.')
	parser.add_argument('--make_undirected', dest='make_undirected', action='store_true',
						help="removes low variance elements using the cv_thresh from data and saves.")
	parser.add_argument('--weigh_edges_by_acc', dest='weigh_edges_by_acc', action='store_true',
						help='weighs edges in output edgelist by accuracy score')
	parser.add_argument('--outfile', dest='outfile', action='store', default=None,
						help='the base name for the output files. Default is preprocessed.tsv')
	parser.add_argument("--r2_threshold", dest="r2_threshold", default=None, type=str,
						help='by including this, the user supplies a value 0 to 1 to remove all models below that r2 threshold')	
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

	delim = args.delim
	thresholds = args.threshold
	rep_path = args.rep_path
	make_undirected = args.make_undirected
	weigh_by_acc = args.weigh_edges_by_acc
	outfile = args.outfile
	r2_thresholds = args.r2_threshold
	verbose = args.verbose

	
	r2_thresholds = [0] if r2_thresholds is None else list(map(float, r2_thresholds.split(',')))
	thresholds = list(map(float, thresholds.split(",")))

        #read output file (either edgelist or output from process)
		# create edgelist from file 
	if verbose:
		print("Reading dataframe", flush=True)
	df = read_dataframe(infile, sep=delim, index_col=0, header=0)
	if verbose:
		print(df.head(), flush=True)
	# create network edgelist
	print("R2 thresholds", r2_thresholds)
	for r2_threshold in r2_thresholds: 
		if verbose: 
			print("Thresholding Models below r2 of: ", r2_threshold)
			print(df.head())
			print(type(r2_threshold)) 
		df = df[ df['r2'] >= r2_threshold ]
		#print(df)
		if verbose:
			print("Creating edgelist", flush=True)
		network_edgelist = create_edgelist( df [ ~df.feature_imps.isna() ], weigh_by_acc)

		#del(df)
	# sort the edgelist 
		if verbose: 
			print("Sorting Edgelist", flush=True)
		sorted_edgelist = sort_edgelist(network_edgelist, 'weight')
		del(network_edgelist)
		if make_undirected: 
			if verbose: 
				print("Collapsing edges to undirected", flush=True)
				print(sorted_edgelist, flush=True)
			sorted_edgelist = transform_edgelist_to_undirected(sorted_edgelist)
		
		print(sorted_edgelist, flush=True)
	# threshold the network 

		for threshold in thresholds: 

			if verbose: 
				print("Thresholding edgelist", flush=True)
			thresholded_edgelist = threshold_edgelist(sorted_edgelist, threshold)
	
		# add in correlates
			if rep_path is not None: 
				if verbose: 
					print("Adding back correlates", flush=True)
				thresholded_edgelist = add_correlates_back_to_df(thresholded_edgelist, rep_path)
			print(thresholded_edgelist, flush=True)
			if outfile is None: 
				outfile_suffix = ""
				if make_undirected:
					outfile_suffix = "_undirected"
				if r2_threshold is not None: 
					outfile_suffix = f"_r2thresholded{r2_threshold}{outfile_suffix}"
				if weigh_by_acc: 
					outfile_suffix = f"_weighted_by_r2{outfile_suffix}"
				outfile = f"network_edgelist_top_{threshold}{outfile_suffix}.tsv"
			thresholded_edgelist.to_csv(outfile, sep='\t',index=False)
			outfile = None
	

if __name__ == "__main__": 
	main()
