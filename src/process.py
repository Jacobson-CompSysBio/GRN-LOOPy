from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
import pandas as pd
import os
import sys
import random
import time
from utils.file_helpers import read_dataframe
from processing.data_helpers import get_train_test_split
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
	parser.add_argument('--device', dest='device', action='store', default="cpu",
						help='the device type you wish to use for training')
	parser.add_argument('--use_mpi', dest='use_mpi', action='store_true',
						help='uses mpi scheduler to run data')
	parser.add_argument('--outfile', dest='outfile', action='store', default='preprocessed.tsv',
						help='the base name for the output files. Default is preprocessed.tsv')
	parser.add_argument("--boosting_type", dest="boosting_type", default  = 'gbdt', 
						help="gbdt is commong gradient boosted tree. gbdt, rf, dart")
	parser.add_argument("--objective", dest="objective", default  = 'regression',
						help="Objective for model: 'regression', 'mape', etc.")						
	parser.add_argument("--learning_rate", dest="learning_rate", default  = 0.1, 
						help="Boosting learning rate")						
	parser.add_argument("--n_estimators", dest="n_estimators", default  = 100,
						help="The total number of leaves allowed in each tree. Recommended set large and fine tune with early stopping on validation")						
	parser.add_argument("--num_leaves", dest="num_leaves", default  = 31,
						help="max number of leaves in one tree")						
	parser.add_argument("--max_depth", dest="max_depth", default  = -1, 
						help="The allowed max depth of the trees. Default is infinite, but suggested to constrain max tree depth to prevent overfitting") # , ")						
	parser.add_argument("--random_state", dest="random_state", default = 42)
	parser.add_argument('--verbose', dest='verbose', action='store_true',
						help='prints verbosely')
	# parser.add_argument("--verbose", dest="verbose", default = 1, help=' -1 = silent, 0 = warn, 1 = info')						

	return parser.parse_args()

train_df = None
test_df = None
device = None 
boosting_type = None # 'gbdt', # 'gbdt'
objective = None # 'regression', # 'regression', 'mape'
learning_rate = None # 0.1, # boosting learning rate
n_estimators = None # 100, # set large and fine tune with early stopping on validation
num_leaves = None # 31, # max number of leaves in one tree
max_depth = None # -1, # constrain max tree depth to prevent overfitting, 
verbose= None # 1, # -1 = silent, 0 = warn, 1 = info
random_state= None # 42,
device = None # "cpu",

def run_mpi_model(feature_name):
	x_cols = train_df.columns[train_df.columns != feature_name]
	y_col = feature_name
	x_train = train_df[x_cols]
	x_test = test_df[x_cols]
	y_train = train_df[y_col]
	y_test = test_df[y_col]


	rank = MPI.COMM_WORLD.Get_rank()
	node_id = os.environ['SLURM_NODEID']
	gpu_device_id = rank % gpus_per_device if device == 'gpu' else -1 


	model = create_model(
		boosting_type = boosting_type, # 'gbdt'
		objective = objective, # 'regression', 'mape'
		learning_rate = learning_rate, # boosting learning rate
		n_estimators = n_estimators, # set large and fine tune with early stopping on validation
		num_leaves = num_leaves, # max number of leaves in one tree
		max_depth = max_depth, # constrain max tree depth to prevent overfitting, 
		random_state= 42,
		device = device,
		gpu_device_id = gpu_device_id, 
		verbose= verbose # -1 = silent, 0 = warn, 1 = info
	)

	print('WE DID IT!')





def main():
	args = get_arguments()

	df_filepath = args.infile
	df = read_dataframe(df_filepath)

	device = args.device
	boosting_type = args.boosting_type
	objective = args.objective
	learning_rate = args.learning_rate
	n_estimators = args.n_estimators
	num_leaves = args.num_leaves
	max_depth = args.max_depth
	random_state = args.random_state
	device = args.device
	verbose = args.verbose


	features = df.columns
	train, test = get_train_test_split(df)
	global train_df
	global test_df
	train_df = train
	test_df = test

	with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
		if executor is not None:
			collected_output = list(executor.map(run_mpi_model, features))

			pd.DataFrame(collected_output).to_csv("this_is_a_test")


if __name__ == '__main__':
	main()