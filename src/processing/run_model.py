import os
import random
import time
from sklearn import metrics

def run_correlation_modeling(model, data, x_cols, y_col, model_type): 
    correlation_scores = np.zeros(len(x_cols))
    start = time.time()
    for col_idx in range(len(x_cols)):
        out = model(data[x_cols[col_idx]],data[y_col])
        if model_type == 'xicor': 
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

def run_rf_model(model, train, test, x_cols, y_col, eval_set=False, device='cpu', model_name = "lgbm", calc_permutation_importance = False, calc_permutation_score=False, n_permutations=1000):
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


    

    feature_importances = None if model_name == 'svr' else "|".join(map(lambda x: f"{x}", model.feature_importances_))
    perm_test_results_p_val = None if not calc_permutation_score else permutation_test_score(model, x_tr, y_tr, n_permutations=n_permutations)[2] # 3rd value is the p_value
    permutation_importances = None if not calc_permutation_importance else "|".join(
        map(lambda x: f"{x}", permutation_importance(model, x_tr, y_tr, n_repeats=n_permutations).importances_mean )
    )

    return {
        'device': device,
        'train_time': stop - start,
        'r2': None if test.shape[1] == 0 else metrics.r2_score( model.predict(x_test), y_test),
        'feature_imps': feature_importances,
        'features': '|'.join(map(lambda x: f"{x}", x_cols)),
        'n_permutations': n_permutations,
        'p_value': perm_test_results_p_val,
        'permutation_importance': permutation_importances,
        'eval_set': eval_set
    }
