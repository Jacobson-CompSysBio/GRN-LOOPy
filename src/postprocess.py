from utils.file_helpers import read_dataframe

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
	df = read_dataframe(df_filepath, sep=delim, header=header_idx)

	# threshold the network 
	

	# add in correlates

	# make undirected if desired. 


if __name__ == "__main__": 
	main()