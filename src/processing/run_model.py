import os
import random
import time
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor


def run_model(model, train, test, x_cols, y_col, eval_set=False, device='cpu', gpus_per_device=8):
    print("splitting data", flush=True)
    x_train = train[x_cols]
    y_train = train[y_col]
    x_test = test[x_cols]
    y_test = test[y_col]

    rank = MPI.COMM_WORLD.Get_rank()
    node_id = os.environ['SLURM_NODEID']
    gpu_device_id = rank % gpus_per_device if device == 'gpu' else -1 

    print("Starting", 
    "\trank:", rank,
    "\tnode_id:", node_id,
    "\tgpu_device_id:", gpu_device_id,
    flush=True)
    start = time.time()
    if eval_set:
        model.fit(x_train, y_train, eval_set=(x_test, y_test))
    else:
        model.fit(x_train, y_train)
    stop = time.time()
    total = stop - start
    prediction = model.predict(x_test)
    r2 = r2_score(prediction, y_test)
    imps = f"{model.feature_importances_}"
    return {
        'device': device,
        'rank': rank,
        'node_id': node_id,
        'gpu_device_id': gpu_device_id,
        'n_jobs': n_jobs,
        'train_time': total,
        'r2': r2,
        'feature_imps': imps,
        'eval_set': eval_set,
    }


