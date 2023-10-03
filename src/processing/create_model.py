import lightgbm as lgb

def create_model(
	boosting_type = 'gbdt', # 'gbdt'
	objective = 'regression', # 'regression', 'mape'
	learning_rate = 0.1, # boosting learning rate
	n_estimators = 100, # set large and fine tune with early stopping on validation
	num_leaves = 31, # max number of leaves in one tree
	max_depth = -1, # constrain max tree depth to prevent overfitting, 
	random_state= 42,
	device = "cpu",
	gpu_device_id = -1,
	min_data_in_leaf = 1,
	verbose= 1 # -1 = silent, 0 = warn, 1 = info
	):
	"""
	TODO: specifically for lgbm, but this function ought to return any type of 
	tree based model that we can run embarassingly to get feature importances
	"""
	kwargs = {}
	if device == "gpu": 
		kwargs['gpu_device_id'] = gpu_device_id
	model = lgb.LGBMRegressor(
		boosting_type= boosting_type,
		objective= objective,
		learning_rate= learning_rate,
		n_estimators= n_estimators,
		num_leaves= num_leaves,
		max_depth= max_depth,
		verbose= verbose,
		random_state= random_state,
		device=device, 
		**kwargs
	)
	return model
