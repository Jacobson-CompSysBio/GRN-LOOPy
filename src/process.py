#!/lustre/orion/world-shared/syb111/frontier_hack/hp_gin_gen_env/bin/python
import pandas as pd
import os
import sys
import random
import time
from utils.file_helpers import read_dataframe
from processing.create_model import AbstractModel
from processing.data_helpers import get_train_test_split
from processing.hyperparam_tuning import hyperparameter_tune
import argparse
import multiprocessing as mp

def get_arguments():
	"""
	Extracts command line arguments. 
	"""
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--infile', type=str, dest='infile', required=True,
						help='the base input file dataframe')
	parser.add_argument('--has_indices', dest='has_indices', action='store_true',
					   help='signifies that dataset does not have indices')
	parser.add_argument('--header_row_idx', dest='header_idx', action='store', default=None,
						help='signifies a header row index for the input file.\ndefault=None\nrows are 0 indexed.')
	parser.add_argument('--delim', dest='delim', action='store', default='\t',
						help='specifies the delimiter for the input file.\n Default "\\t"')
	parser.add_argument('--device', dest='device', action='store', default="cpu",
						help='the device type you wish to use for training')
	parser.add_argument('--mpi', dest='mpi', action='store_true', default=False,
						help='the flag for determining whether or not MPI is used')
	parser.add_argument('--outfile', dest='outfile', action='store', default='processed_data.tsv',
						help='the base name for the output files. Default is processed_data.tsv')
	parser.add_argument("--boosting_type", dest="boosting_type", default  = 'gbdt', 
						help="gbdt is commong gradient boosted tree. gbdt, rf, dart")
	parser.add_argument("--model_name", dest="model_name", default  = 'lgbm',
						help="Objective for model: 'regression', 'mape', etc.")						
	parser.add_argument("--objective", dest="objective", default  = 'regression',
						help="Objective for model: 'regression', 'mape', etc.")						
	parser.add_argument("--learning_rate", dest="learning_rate", default  = 0.1, 
						help="Boosting learning rate")						
	parser.add_argument("--n_estimators", dest="n_estimators", default  = 100, type=int,
						help="The total number of leaves allowed in each tree. Recommended set large and fine tune with early stopping on validation")						
	parser.add_argument("--num_leaves", dest="num_leaves", default  = 31,
						help="max number of leaves in one tree")						
	parser.add_argument("--max_depth", dest="max_depth", default  = -1, 
						help="The allowed max depth of the trees. Default is infinite, but suggested to constrain max tree depth to prevent overfitting") # , ")						
	parser.add_argument("--random_state", dest="random_state", default = 42)
	parser.add_argument("--calc_permutation_importance", dest="calc_permutation_importance", action="store_true", 
						help="A flag defining whether or not permutation importance scores will be generated for X features within each model"),
	parser.add_argument("--calc_permutation_score", dest="calc_permutation_score", action="store_true", 
						help="A flag defining whether or not an overal permutation test score will be generated for each model"),
	parser.add_argument("--n_permutations", dest="n_permutations", action="store", type=int, default=1000, 
						help="The total number of permutations defined for the permutation importance and permutation score."),
	parser.add_argument("--verbose", dest="verbose", default = 1, help=' -1 = silent, 0 = warn, 1 = info')						
	parser.add_argument("--bagging_fraction", dest="bagging_fraction", default=None,
						help="The total fraction of samples in the training set to be bagged")
	parser.add_argument("--bagging_freq", dest="bagging_freq", default=1, 
						help="The frequency to undergo bagging. Default is 1, for every iteration.")
	parser.add_argument("--tune_hyper_parameters", dest="tune_hyper_parameters", action="store_true", 
						help="engage in hyperparameter tuning for each feature.")
	parser.add_argument("--n_processes", dest="n_processes", action="store", default=8, type=int,
						help="the total number of processes to each model")
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
verbose= -1 # 1, # -1 = silent, 0 = warn, 1 = info
random_state= None # 42,
model_name=None
calc_permutation_importance = None
calc_permutation_score = None
n_permutations = None
bagging_fraction=None
bagging_freq=None
tune_hyper_parameters = None
n_processes = None 
mpi = None

def get_model_hyper(): 
	if mpi: 
		from mpi4py import MPI

	model_hyper = None
	model_fixed = None
	if model_name == 'lgbm': 
		model_hyper = {
			'learning_rate': [0.5, 0.1, 0.05],#, 0.001],
			'min_split_gain': [0, 0.05, 0.1],#, 0.25, 0.5],
			'min_data_in_leaf': [1, 5, 10] #, 20],
	 	}
		model_fixed = {
			'model_name': 'lgbm',
			'max_depth': 13,#, 25, 50, 100],
			'num_leaves': 15,#, 31, 63],#, 127, 255],
			'objective': 'regression',
			'device': 'cpu',
			'verbose': -1, 
			'early_stopping_round': 15
		}
	if model_name == 'rf': 
		model_hyper = {
			'n_estimators': [50, 100, 250],
			'max_depth': [25, 50, 100],#, 1000],
			'num_leaves': [15, 31, 63],
			'bagging_fraction': [0.25, 0.5, 0.75]
 		}
		model_fixed = {
			'model_name': 'rf',
			'objective': 'regression',
			'device': 'cpu',
			'bagging_freq': 1,
			'verbose': -1,
			'early_stopping_round': 15
		}
	return model_hyper, model_fixed

def run_mpi_model(feature_name):
	if mpi: 
		from mpi4py import MPI

	print("FEATURE NAME", feature_name)
	print("traindf ", train_df)
	x_cols = train_df.columns[train_df.columns != feature_name]
	y_col = feature_name

	print(x_cols)
	print(y_col)


    #TODO:  run w/ salloc/srun to limit the gpus that the process sees. 
	rank = "no_mpi" if mpi is None else MPI.COMM_WORLD.Get_rank()
	node_id = "no_mpi" if mpi is None else os.environ['SLURM_NODEID']
	#gpus_per_device = 8
	gpu_device_id = 0 #rank % gpus_per_device if device == 'gpu' else -1 
	n_jobs = -1

	#TODO: if model does not use GPUs, ensure to warn user and switch 
	# back to CPU specifications. 
	param_list = [{
		"learning_rate": learning_rate,
		"n_estimators": n_estimators,
		"max_depth": max_depth,
		"bagging_fraction": bagging_fraction,
		"bagging_freq": bagging_freq,
	}]

	if model_name == 'irf': 
		param_list[0]['n_jobs'] = n_jobs
		param_list[0]['n_iterations'] = 5

	if tune_hyper_parameters: 
		print("hyperparam tuning")
		model_hyper, model_fixed = get_model_hyper()
		param_list = hyperparameter_tune(
			model_class = AbstractModel,
			model_hyper = model_hyper,
			model_fixed = model_fixed,
			train_hyper = {},
			train_fixed = {},
			data = train_df,
			y_feature= y_col,
			k_folds= 5,
			train_size=0.85,
			device=device,
			verbose= False,
		)
	print("Creating abstract model")
	model = AbstractModel(
		model_name=model_name,
		objective=objective,
		model_type=objective,
		device=device,
		gpu_device_id=gpu_device_id,
		verbose=-1,
		**param_list[0]
	)
	output = model.fit(train_df,
		test_df,
		x_cols,
		y_col,
		calc_permutation_importance = calc_permutation_importance,
		calc_permutation_score = calc_permutation_score,
		n_permutations = n_permutations,
		eval_set=True)
	
	output['rank']= rank
	output['node_id']= node_id
	output['gpu_device_id']= gpu_device_id
	output['n_jobs']= n_jobs
	output['feature'] = feature_name

	return output



def main():
	print("getting arguments")
	args = get_arguments()

	global train_df
	global test_df
	global device
	global boosting_type
	global objective
	global model_name
	global learning_rate
	global n_estimators
	global num_leaves
	global max_depth
	global random_state
	global device
	global calc_permutation_importance
	global calc_permutation_score
	global n_permutations
	global bagging_fraction
	global bagging_freq
	global tune_hyper_parameters
	global verbose
	global mpi
	global n_processes

	df_filepath = args.infile
	has_indices = args.has_indices
	header_idx = int(args.header_idx)
	delim = args.delim 

	mpi = args.mpi
	device = args.device
	boosting_type = args.boosting_type
	objective = args.objective
	learning_rate = args.learning_rate
	n_estimators = args.n_estimators
	num_leaves = args.num_leaves
	max_depth = args.max_depth
	random_state = args.random_state
	calc_permutation_importance = args.calc_permutation_importance
	calc_permutation_score = args.calc_permutation_score
	n_permutations = args.n_permutations
	model_name = args.model_name
	bagging_fraction = args.bagging_fraction
	bagging_freq = args.bagging_freq
	tune_hyper_parameters = args.tune_hyper_parameters
	n_processes = args.n_processes
	verbose = args.verbose
	outfile = args.outfile



	if mpi: 
		from mpi4py.futures import MPICommExecutor
		from mpi4py import MPI





	print("Reading Data In")
	df = read_dataframe(df_filepath, sep=delim, header=header_idx)
	print(df.columns)
	print("Splitting Data")
	features = df.columns
	train, test = get_train_test_split(df)
	train_df = train
	test_df = test
	print(train_df)
	print(test_df)

	print("Distributing Data")
	if mpi: 
		with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
			if executor is not None:
				collected_output = list(executor.map(run_mpi_model, features))
				pd.DataFrame(collected_output).to_csv(outfile, sep=delim)
				# print(collected_output)
	else: 
		with mp.Pool(processes=n_processes) as pool:
        # Map tasks to workers
			collected_output = list(pool.map(run_mpi_model, features))
    

## TODO: We need to factor in the kfold crass validation. Additionally, for data sets too small, a functionality should be implemented for folks to forego the validation step in the train/test split (for astoundingly small rna seq data sets)
if __name__ == '__main__':
	main()
