import os
import random
from sklearn import metrics
import time
from .create_model import create_model
from .data_helpers import get_train_test_split


def hyper_parameter_tune(
	data,
	x_cols,
	y_col,
	k=5,
	boosting_type = 'gbdt',
	objective = 'regression',
	n_estimators = 500,
	random_state= 42,
	device = 'cpu',
	max_depth = -1,
	gpu_device_id = -1,
	verbose=-1):
	"""
	This hyperparameter tuning function tunes the parameters using a basic grid search: 
	"""
	learning_rate_list = [0.1, 0.05, 0.01]
	num_leaves_list = [15,31,45]
	min_data_in_leaf_list = [0.03, 0.01]
	best_model_estimation= None
	model_list = []
	for learning_rate in learning_rate_list: 
		for num_leaves in num_leaves_list: 
			for min_leaf in min_data_in_leaf_list: 
				for fold in range(k): 
					train, test = get_train_test_split(data)
					x_train = train[x_cols].to_numpy()
					y_train = train[y_col].to_numpy()
					x_test = test[x_cols].to_numpy()
					y_test = test[y_col].to_numpy()
					min_leaf_count = int(len(y_train) * min_leaf)
					model = create_model(
						boosting_type = boosting_type,
						objective = objective,
						n_estimators=n_estimators,
						learning_rate=learning_rate, 
						num_leaves=num_leaves,
						min_data_in_leaf = min_leaf_count,
						device=device,
						gpu_device_id = gpu_device_id,
						verbose=verbose
					)
					model.fit(x_train, y_train, eval_set=(x_test, y_test))
					prediction = model.predict(x_test)
					r2 = metrics.r2_score(prediction, y_test)
					l2_by_estimator = model.evals_result_['valid_0']['l2']
					lowest_n_iter_l2_idx = np.argmin(l2_by_estimator)
					lowest_l2_value = l2_by_estimator[lowest_n_iter_l2_idx]
					model_params = {
						"learning_rate": learning_rate, 
						"num_leaves": num_leaves,
						"min_leaf": min_leaf, 
						"estimator": lowest_n_iter_l2_idx,
						"lowest_l2_value": lowest_l2_value,
						"r2": r2,
						"fold": fold
					}
					model_list.append(model_params)
	return pd.DataFrame(model_list).sort_values(by='lowest_l2_value')

def run_and_doc_model(model, train, test, x_cols, y_col, eval_set=False, device='cpu'):
	x_train = train[x_cols].to_numpy()
	y_train = train[y_col].to_numpy()
	x_test = test[x_cols].to_numpy()
	y_test = test[y_col].to_numpy()
	#train
	start = time.time()
	if eval_set:
		model.fit(x_train, y_train, eval_set=(x_test, y_test))
	else:
		model.fit(x_train, y_train)
	stop = time.time()
	total = stop - start
	prediction = model.predict(x_test)
	r2 = metrics.r2_score(prediction, y_test)
	#imps
	imps = "|".join(map(lambda x: f"{x}", model.feature_importances_))
	features = "|".join(map(lambda x: f"{x}", x_cols))
	#return
	return {
		'device': device,
		'train_time': total,
		'r2': r2,
		'feature_imps': imps,
		'features': features,
		'eval_set': eval_set
	}


