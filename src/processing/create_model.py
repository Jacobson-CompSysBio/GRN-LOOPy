from irf.ensemble import RandomForestRegressorWithWeights
from sklearn.inspection import permutation_importance
from sklearn.model_selection import permutation_test_score
from sklearn import metrics
import numpy as np
import time



class AbstractModel: 
    """
    This is a wrapper class for the ML / statistical models for network generation. 
    """
    model_name = None
    objective = None
    model = None

    def __init__(self, model_name: str, objective: str, **kwargs):
        self.model_name = model_name
        self.objective = objective
        self.device = kwargs['device']
        self.define_model(**kwargs)

    def define_model(self, **kwargs):
        self.model = RandomForestRegressorWithWeights(
            n_estimators=kwargs['n_estimators'],
            criterion='mse',
            max_depth=kwargs['max_depth'],
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None
        )

    def run_rf_model(self, train, test, x_cols, y_col, n_iterations, eval_set=False, device='cpu', model_name = "lgbm", calc_permutation_importance = False, calc_permutation_score=False, n_permutations=1000):
        x_train = train[x_cols]
        y_train = train[y_col]
        x_test = test[x_cols]
        y_test = test[y_col]
    

        start = time.time()
        
        feature_weights = np.ones(x_train.shape[1])
        weight = feature_weights / np.sum(feature_weights)
        for i in range(n_iterations): 
            print('running iteration ', i)
            self.model.fit(x_train, y_train, feature_weight = weight)
            feature_imps = self.model.feature_importances_
            weight = feature_imps / sum(feature_imps)

        stop = time.time()
        print("calc_permutation_score", calc_permutation_score)
        print("calc_permutation_importance", calc_permutation_importance)
        feature_importances = None if model_name == 'svr' else "|".join(
            map(lambda x: f"{x}", self.model.feature_importances_))
        perm_test_results_p_val = None if not calc_permutation_score else permutation_test_score(
            self.model, x_train, y_train, n_permutations=n_permutations)[2] # 3rd value is the p_value
        permutation_importances = None if not calc_permutation_importance else "|".join(
            map(lambda x: f"{x}", permutation_importance(self.model, x_train, y_train, n_repeats=n_permutations).importances_mean )
        )
    
        return {
            'device': device,
            'train_time': stop - start,
            'r2': None if test.shape[1] == 0 else metrics.r2_score( self.model.predict(x_test), y_test ),
            'feature_imps': feature_importances,
            'features': '|'.join(map(lambda x: f"{x}", x_cols)),
            'n_permutations': n_permutations,
            'p_value': perm_test_results_p_val,
            'permutation_importance': permutation_importances,
            'eval_set': eval_set
        }


    
    def fit(self, train, test, x_cols, y_col, n_iterations=5, eval_set=None,  calc_permutation_importance = False, calc_permutation_score = False, n_permutations = 1000):
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

        return return_data
