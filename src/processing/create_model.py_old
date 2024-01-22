import lightgbm as lgb
from sklearn.inspection import permutation_importance
from sklearn.model_selection import permutation_test_score
from scipy import stats
from sklearn import metrics
from sklearn.svm import SVR
from xicor import xicor
import time
from lightgbm import log_evaluation
import numpy as np

def create_lgb_model(
    boosting_type = 'gbdt', # 'gbdt'
    objective = 'regression', # 'regression', 'mape'
    learning_rate = 0.1, # boosting learning rate
    n_estimators = 100, # set large and fine tune with early stopping on validation
    num_leaves = 31, # max number of leaves in one tree
    max_depth = -1, # constrain max tree depth to prevent overfitting, 
    random_state= 42,
    device = "cpu",
    gpu_device_id = -1,
    verbose= 1,
    **kwargs):  # -1 = silent, 0 = warn, 1 = info
    """
    TODO: specifically for lgbm, but this function ought to return any type of 
    tree based model that we can run embarassingly to get feature importances
    """
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
 

class AbstractModel: 
    """
    This is a wrapper class for the ML / statistical models for network generation. 
    """
    model_name = None
    objective = None
    model = None
    device = None

    train_time = None
    r2 = None
    feature_imps = None
    features = None
    n_permutations = None
    p_value = None
    permutation_importance = None
    eval_set = None

    best_iteration = None
    best_score = None
    
    def __init__(self, model_name: str, objective: str, **kwargs):
        self.model_name = model_name
        self.objective = objective
        self.device = kwargs['device']
        self.define_model(**kwargs)
    def define_model(self, **kwargs):
        if self.model_name == 'lgbm': 
            self.model = create_lgb_model(
                boosting_type = 'gbdt',
                objective = self.objective,
                **kwargs
            )
        if self.model_name == 'rf':
            ## ABSOLUTELY MUST DEFINE bagging_freq and
            ## bagging_frac. 
            ##  bagging_freq=1,
            ##  bagging_fraction=0.6, 
            self.model = create_lgb_model(
                boosting_type = 'rf',
                objective = self.objective,
                **kwargs
            )
        if self.model_name == 'dart':
            ## ABSOLUTELY MUST DEFINE bagging_freq and
            ## bagging_frac. 
            ##  bagging_freq=1,
            ##  bagging_fraction=0.6, 
            self.model = create_lgb_model(
                boosting_type = 'dart',
                objective = self.objective,
                **kwargs
            )
        if self.model_name == 'svr':
            self.model = SVR()
        if self.model_name == 'pearson':
            self.model = stats.pearsonr
        if self.model_name == 'spearman': 
            self.model = stats.spearmanr
        if self.model_name == 'xicor': 
            self.model = xicor.Xi
    def run_correlation_modeling(self, data, x_cols, y_col): 
        correlation_scores = np.zeros(len(x_cols))
        start = time.time()
        for col_idx in range(len(x_cols)):
            out = self.model(data[x_cols[col_idx]],data[y_col])
            if self.model_name == 'xicor': 
                correlation_scores[col_idx] = out.correlation
            else: 
                correlation_scores[col_idx] = out[0]
        stop = time.time()
        return {
            "device": "cpu",
            "train_time": stop-start,
            "features": "|".join(map(lambda x: f"{x}", x_cols)),
            "feature_cors": "|".join(map(lambda x: f"{x}", correlation_scores))
        }
    def run_rf_model(self, train, test, x_cols, y_col, eval_set=False, device='cpu', model_name = "lgbm", calc_permutation_importance = False, calc_permutation_score=False, n_permutations=1000):
        x_train = train[x_cols]
        y_train = train[y_col]
        x_test = test[x_cols]
        y_test = test[y_col]
        start = time.time()
        if eval_set:
            callbacks = [log_evaluation(-1)]
            self.model.fit(x_train, y_train, eval_set=(x_test, y_test), callbacks=callbacks )
        else:
            self.model.fit(x_train, y_train)
        stop = time.time()
        feature_importances = None if model_name == 'svr' else "|".join(
            map(lambda x: f"{x}", self.model.feature_importances_))
        permutation_test_start = None if not calc_permutation_score else time.time()
        perm_test_results_p_val = None if not calc_permutation_score else permutation_test_score(
            self.model, x_train, y_train, n_permutations=n_permutations)[2] # 3rd value is the p_value
        permutation_test__stop = None if not calc_permutation_score else time.time()
        permutation_importance_start = None if not calc_permutation_score else time.time()
        permutation_importances = None if not calc_permutation_importance else "|".join(
            map(lambda x: f"{x}", permutation_importance(self.model, x_train, y_train, n_repeats=n_permutations).importances_mean )
        )
        permutation_importance_stop = None if not calc_permutation_score else time.time()
        
        self.train_time = stop - start
        self.r2 =  None if test.shape[1] == 0 else metrics.r2_score( self.model.predict(x_test), y_test ),
        self.feature_imps = feature_importances
        self.features = '|'.join(map(lambda x: f"{x}", x_cols))
        self.n_permutations = n_permutations
        self.p_value = perm_test_results_p_val
        self.permutation_importance = permutation_importances
        self.eval_set = eval_set
        
        best_score, best_iteration = self.evaluate()
        self.best_iteration = best_iteration
        self.best_score = best_score
        
        
        return {
            'device': self.device,
            'train_time': self.train_time,
            'r2': self.r2,
            'feature_imps': self.feature_imps,
            'features': self.features,
            'n_permutations': self.n_permutations,
            'p_value': self.p_value,
            'permutation_importance': self.permutation_importance,
            'eval_set': self.eval_set
        }
    def fit(self, train, test, x_cols, y_col, eval_set=None,  calc_permutation_importance = False, calc_permutation_score = False, n_permutations = 1000):
        return_data = None
        if self.objective != 'correlation': 
            return_data = self.run_rf_model(
                train = train,
                test = test,
                x_cols = x_cols,
                y_col = y_col,
                eval_set = eval_set,
                device = self.device,
                model_name = self.model_name,
                calc_permutation_importance = calc_permutation_importance,
                calc_permutation_score = calc_permutation_score,
                n_permutations = n_permutations
            )
        if self.objective == 'correlation':
            return_data = self.run_correlation_modeling(
                pd.concat([train, test]), 
                x_cols,
                y_col,
                self.model_name
            )   
        return return_data
    def evaluate(self):
        if self.model_name == 'svr': 
            return self.r2, None
        print('MODEL?', self.model.evals_result_)
        score_idx = np.argmin(self.model.evals_result_['valid_0']['l2'])
        
        return self.model.evals_result_['valid_0']['l2'][score_idx], score_idx
        
