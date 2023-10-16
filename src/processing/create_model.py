import lightgbm as lgb
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from xicor import xicor

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

    def __init__(self, model_name: str, objective: str, **kwargs):
        self.model_name = model_name
        self.objective = objective
        self.device = kwargs['device']
        self.define_model(**kwargs)

    def define_model(self, **kwargs):
        if self.model_name == 'lgbm': 
            self.model = create_lgb_model(
                boosting_type = 'gbdt',
                **kwargs
            )
        if self.model_name == 'rf':
            ## ABSOLUTELY MUST DEFINE bagging_freq and
            ## bagging_frac. 
            ##  bagging_freq=1,
            ##  bagging_fraction=0.6, 
            self.model = create_lgb_model(
                boosting_type = 'rf',
                **kwargs
            )

        if self.model_name == 'dart':
            ## ABSOLUTELY MUST DEFINE bagging_freq and
            ## bagging_frac. 
            ##  bagging_freq=1,
            ##  bagging_fraction=0.6, 
            self.model = create_lgb_model(
                boosting_type = 'dart',
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
    
    def run_correlation_modeling(self, data, x_cols, y_col, objective): 
        correlation_scores = np.zeros(len(x_cols))
        start = time.time()
        for col_idx in range(len(x_cols)):
            out = self.model(data[x_cols[col_idx]],data[y_col])
            if objective == 'xicor': 
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
            self.model.fit(x_train, y_train, eval_set=(x_test, y_test))
        else:
            self.model.fit(x_train, y_train)
        stop = time.time()
    
    
        
    
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