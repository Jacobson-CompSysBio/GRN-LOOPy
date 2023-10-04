import pandas as pd
from sklearn import datasets 
from sklearn import model_selection

def get_train_test_split(df = None,
						df_train = None,
						df_test = None,
						feature_name = None,
						n_samples = None,
						n_features=None,
						log_transform=False): 
	"""
	If a user supplies a dataframe, then the dataframe will be partitioned into a train
	test split assuming that the user provides a feature name. 

	If no base dataframe is included, but instead a train and test dataframe are
	supplied, those two will be partitioned into x_train/x_test/y_train/y_test 

	If no feature name i`s included, an error will be thrown. 

	If no base data are provided, then a regression model from sklearn's "make regression" 
	will be created, assuming the user provided the number of samples and features. 
	"""
	if (df is None and df_train is None and df_test is None) :
		if n_samples is None or n_features is None: 
			raise(Exception("Both samples and features must be included."))
		x, y = datasets.make_regression(n_samples, n_features)
		df = pd.DataFrame(x)
		df[n_features] = y
		feature_name = n_features
	if (df is not None): 
		df_train, df_test = model_selection.train_test_split(df)
		if feature_name is None: 
			return df_train, df_test
	if (df_train is not None and df_test is not None): 
		print("Separating to x,y train/test split.")
		if feature_name is None: 
			raise(Exception("No feature name for dataframes"))
		x_cols = list(filter(lambda x: x != feature_name, df_train.columns))
		y_col = feature_name
		if log_transform: 
			df_train
		x_train = df_train[x_cols]
		x_test = df_test[x_cols]
		y_train = df_train[y_col]
		y_test = df_test[y_col]
		return(x_train, x_test, y_train, y_test)
	raise(Exception("No other viable options, please read documentation"))

