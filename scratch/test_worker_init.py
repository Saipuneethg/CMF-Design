import os
import time
import numpy as np
from multiprocessing import Pool

# Worker loads its own data from disk or memory
_W_DATA = None

def _init_worker():
    global _W_DATA
    # Just load small slice or compute
    _W_DATA = np.ones((40000, 150), dtype=np.float32)

def _worker_task(x):
    return float(np.sum(_W_DATA[:1000, :10] * x))

if __name__ == "__main__":
    t0 = time.time()
    with Pool(processes=8, initializer=_init_worker) as pool:
        res = pool.map(_worker_task, [i * 0.1 for i in range(50)])
    print("Success! Time:", time.time() - t0)
