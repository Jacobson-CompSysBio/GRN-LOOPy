from irf.ensemble import RandomForestClassifierWithWeights, RandomForestRegressorWithWeights
from sklearn.inspection import permutation_importance
from sklearn.model_selection import permutation_test_score
from sklearn import metrics
import numpy as np
import time
from lightgbm import log_evaluation
import numpy as np



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
        if self.model_name == 'irf': 
            self.model = RandomForestRegressorWithWeights(
                n_estimators=kwargs['n_estimators'],
                max_depth=kwargs['max_depth'],
                n_jobs=kwargs['n_jobs'],
                criterion='squared_error',
                min_weight_fraction_leaf=0.0,
                max_features='sqrt', #kwargs['max_features'], TODO: Matt & Alice explore parameterization of this
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=True,
                verbose=0
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

    def fit_lgb_rf_model(): 
        if eval_set:
            callbacks = [log_evaluation(-1)]
            self.model.fit(x_train, y_train, eval_set=(x_test, y_test), callbacks=callbacks )
        else:
            self.model.fit(x_train, y_train)

    def fit_irf_model(): 
        feature_weights = np.ones(x_train.shape[1])
        weight = feature_weights / np.sum(feature_weights)
        for i in range(n_iterations):
            self.model.fit(x_train, y_train, feature_weight = weight)
            feature_imps = self.model.feature_importances_
            weight = feature_imps / sum(feature_imps)
            if verbose: 
                print(f"Iteration time @", i, ":  ", time.time() - start) 

    def calculate_pumutation_test(): 
        permutation_test_start = None if not calc_permutation_score else time.time()
        perm_test_results_p_val = None if not calc_permutation_score else permutation_test_score(
            self.model, x_train, y_train, n_permutations=n_permutations)[2] # 3rd value is the p_value
        permutation_test__stop = None if not calc_permutation_score else time.time()

    def calculate_permutation_importance(): 
        permutation_importance_start = None if not calc_permutation_score else time.time()
        permutation_importances = None if not calc_permutation_importance else "|".join(
            map(lambda x: f"{x}", permutation_importance(self.model, x_train, y_train, n_repeats=n_permutations).importances_mean )
        )
        permutation_importance_stop = None if not calc_permutation_score else time.time()

    def run_rf_model(self, train, test, x_cols, y_col, eval_set=False, device='cpu', model_name = "lgbm", calc_permutation_importance = False, calc_permutation_score=False, n_permutations=1000):
        x_train = train[x_cols]
        y_train = train[y_col]
        x_test = test[x_cols]
        y_test = test[y_col]


        start = time.time()
            if self.model_name in ['lgbm', 'rf', 'dart']: 
                run_lgb_rf_model(x_train, y_train, x_test, y_test)
            elif self.model_name == 'irf':
                fit_irf_model
        stop = time.time()


        feature_mask = np.arange(len(x_cols)) if model_name == 'svr' else np.argwhere(self.model.feature_importances_ > 0)
        feature_importances = None if model_name == 'svr' else "|".join(
            map(lambda x: f"{x}", np.array(self.model.feature_importances_)[feature_mask].reshape(-1)))
        feature_names = np.array(x_cols)[feature_mask].reshape(-1)
        
        self.train_time = stop - start
        self.r2 =  self.model.oob_score_ if test.shape[1] == 0 else metrics.r2_score( self.model.predict(x_test), y_test ),
        self.feature_imps = feature_importances
        self.features = feature_names
        self.n_permutations = None if n_permutations

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
                n_iterations = n_iterations,
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
        
