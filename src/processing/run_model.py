import os
import random
import time
from sklearn import metrics

def run_model(model, train, test, x_cols, y_col, eval_set=False, device='cpu'):
	x_train = train[x_cols]
	y_train = train[y_col]
	x_test = test[x_cols]
	y_test = test[y_col]

	start = time.time()
	if eval_set:
		model.fit(x_train, y_train, eval_set=(x_test, y_test))
	else:
		model.fit(x_train, y_train)
	stop = time.time()
	total = stop - start
	prediction = model.predict(x_test)
	r2 = metrics.r2_score(prediction, y_test)

	imps = ",".join(map(lambda x: f"{x}", model.feature_importances_))
	features = ','.join(map(lambda x: f"{x}", x_cols))

	return {
		'device': device,
		'train_time': total,
		'r2': r2,
		'feature_imps': imps,
		'features': features,
		'eval_set': eval_set
	}


