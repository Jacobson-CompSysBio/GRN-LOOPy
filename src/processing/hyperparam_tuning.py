import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def hyperparameter_tune(
    model_class,
    model_hyper: dict,
    model_fixed: dict,
    train_hyper: dict,
    train_fixed: dict,
    data: pd.DataFrame, # Or dataframe?
    y_feature: str,
    k_folds: int,
    train_size: float = 0.85,
    device: str= 'cpu',
    verbose: bool = True,
    ):
    '''
    This function performs hyperparameter tuning for a given model class.

    Note that the model class must have the following methods:
        - fit(X, y, eval_set=(X_val, y_val), **kwargs) -> None
        - evaluate(X, y) -> float

    For example, the model class could be a wrapper for a LightGBM model, 
    where the fit method is already implemented, and the evaluate method
    can be implemented as follows:

        def evaluate(self, X, y):
            score = min(self.evals_result_['valid_0']['binary_logloss'])

    Note that this example applies to a LightGBM classifier, and that the
    evaluate method returns the minimum validation log loss, which was 
    already computed during training.

    Parameters
    ----------
    model_class : class
        Class of model to use for training and evaluation, must include
        fit and evaluate methods, see above.
    model_hyper : dict
        Dictionary of tunable hyperparameters for the model. Each value
        must be a list of hyperparameter values to try.
    model_fixed : dict
        Dictionary of fixed hyperparameters for the model. Each value
        must be a single hyperparameter value.
    train_hyper : dict
        Dictionary of tunable hyperparameters for the training procedure.
        Each value must be a list of hyperparameter values to try.
    train_fixed : dict
        Dictionary of fixed hyperparameters for the training procedure.
        Each value must be a single hyperparameter value.
    data : pd.DataFrame
        Input dataframe to use for training and evaluation [n_samples, n_features].
    y_feature : str
        Target variable name used to identify y vector column
    k_folds : int
        The number of folds k that the user wishes to run each set 
        of parameters
    val_indices : list
        List of index sets to use for validation. Each index set must be
        an iterable of indices (i.e., rows of the dataset) that can be
        used to index numpy arrays.
    weights : list, optional
        List of weights to use for weighted average of scores for each
        hyperparameter combination.
    verbose : bool, optional
        Whether to print results and use status bar during tuning.

    Returns
    -------
    best_model_params : dict
        Dictionary of best model hyperparameters.
    best_train_params : dict
        Dictionary of best training hyperparameters.
    model_list : list
        List of models for each hyperparameter and train/val combination.
    score_list : list
        List of scores for each hyperparameter and train/val combination.
    '''
    if model_fixed['model_name'] in ['pearson', 'spearman', 'xicor']: 
        print(f"Correlation models require no parameter tuning")
        return
    n_samples = data.shape[0]
    train_indices = []
    val_indices = []
    for i in range(k_folds): 
        train, val = train_test_split(range(n_samples), train_size=train_size)
        train_indices.append(train)
        val_indices.append(val)
    x_cols = data.columns[ data.columns != y_feature ] 
    y_col = y_feature
    # unpack hyperparameters
    model_h_keys = list(model_hyper.keys())
    model_h_vals = list(model_hyper.values())
    train_h_keys = list(train_hyper.keys())
    train_h_vals = list(train_hyper.values())
    # get number of hyperparameter combinations
    model_h_lens = [len(h) for h in model_h_vals]
    train_h_lens = [len(h) for h in train_h_vals]
    n_combinations = np.prod(model_h_lens + train_h_lens)
    # initialize list to store model scores
    model_list, score_list, r2_list = {}, {}, {}
    # loop over hyperparameter combinations
    loop = tqdm(range(n_combinations)) if verbose else range(n_combinations)
    for i in loop:
        model_hyper_i = None
        train_hyper_i = None
        try: 
            # unravel hyperparameter values
            indices = np.unravel_index(i, model_h_lens + train_h_lens)
            model_hyper_i = {k: v[i] for k, v, i in zip(
                model_h_keys, 
                model_h_vals, 
                indices[:len(model_h_keys)])}
            train_hyper_i = {k: v[i] for k, v, i in zip(
                train_h_keys, 
                train_h_vals, 
                indices[len(model_h_keys):])}
            # fit models on train/val sets a nd store scores
            models, scores, r2s = [], [], []
            for t_idx, v_idx in zip(train_indices, val_indices):
                # initialize model with hyperparameter combination
                model = model_class(**model_hyper_i, **model_fixed)
                # fit model on train/val set
                model_output = model.fit(
                    train=data.iloc[t_idx],
                    test = data.iloc[v_idx],
                    x_cols=x_cols, 
                    y_col= y_col,
                    eval_set=  True  #if model_fixed['model_name'] in ['lgbm', 'dart', 'rf', 'irf'] else False,
                    # **train_hyper_i,
                    # **train_fixed,
                )
                # evaluate model on val set
                # IF model is lgbm, rf, or dart, the score ought to be minimized,
                # IF using R2, score ought to be maximized :/ 
                # Not ideal.... need a better method of optimization.
                score = model.best_score
                print("SCORE")
                # store model and score for train/val set
                models.append(model)
                scores.append(score)
                r2s.append(model.r2)
            # store scores for hyperparameter combination
            print("MODELS")
            model_list[i] = models
            score_list[i] = scores
            r2_list[i] = r2s
        except Exception as ex: 
            print("EXCEPTION in Hyperparam Tuning:")
            print("Train Hyper", train_hyper_i)
            print("Model Hyper", model_hyper_i)
            print("y_col", y_col)
            print(ex)
            #print("Exception", ex)
            model_list[i] = np.nan
            score_list[i] = np.nan
            r2_list[i] = np.nan
    # compute weighted scores for each hyperparameter combination
    eval_matrix = np.zeros(model_h_lens + train_h_lens)
    for i, scores in zip(score_list.keys(), score_list.values()):
        indices = np.unravel_index(i, model_h_lens + train_h_lens)
        eval_matrix[indices] = np.average(scores)#, weights=weights)
    # get best hyperparameters
    print(eval_matrix)
    
    argmin = np.nanargmin(eval_matrix) if model_fixed['model_name'] != 'svr' else np.nanargmax(eval_matrix)
    indices = np.unravel_index(argmin, eval_matrix.shape)
    best_model_params = {k: v[i] for k, v, i in zip(
        model_h_keys,
        model_h_vals,
        indices[:len(model_h_keys)])}
    best_train_params = {k: v[i] for k, v, i in zip(
        train_h_keys,
        train_h_vals,
        indices[len(model_h_keys):])}
    # print results
    if verbose:
        print()
        print('best model parameters:')
        just = max([len(k) for k in model_h_keys]) if len(model_h_keys) > 0 else 0
        for k, v in best_model_params.items():
            print(f'{k.ljust(just)} {v}')
        print()
        print('best train parameters:')
        just = max([len(k) for k in train_h_keys]) if len(train_h_keys) > 0 else 0
        for k, v in best_train_params.items():
            print(f'{k.ljust(just)} {v}')
        print()
        print('eval score:')
        print(f'score: {eval_matrix[indices]}')

    best_idx = np.argmin(list(map(lambda x: np.mean(x), score_list)))
    r2_variance = list(map(lambda x: np.var(score_list[x]), score_list.keys())) # np.var(score_list)
    
    return {
        "best_model_params": best_model_params, 
        "best_train_params": best_train_params, 
        "model_list": model_list, 
		"model_r2_variance": r2_variance, 
        #"score_list": score_list, 
        #"r2_list": r2_list,
        "best_idx": best_idx, 
        "argmin": argmin
    }
