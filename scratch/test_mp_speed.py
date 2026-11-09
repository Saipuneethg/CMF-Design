import os
import time
import numpy as np
from multiprocessing import Pool

P = 40000
N = 150
H_global = np.random.uniform(0.1, 10.0, size=(P, N))
A_sub_global = np.random.randn(P, 8)

def eval_single(args):
    M, pca_basis = args
    A_theta = M @ pca_basis
    U, S, Vt = np.linalg.svd(A_theta)
    if S[-1] < 1e-4:
        return 999.0
    N_theta = Vt[3:].T
    G = A_sub_global @ N_theta
    n_pts = H_global.shape[1]
    spans = np.zeros((n_pts, 5))
    for j in range(5):
        G_j = G[:, j]
        p_idx = np.where(G_j > 1e-9)[0]
        inv_G_p = (1.0 / G_j[p_idx])[:, None]
        t_p = np.min(H_global[p_idx] * inv_G_p, axis=0)
        n_idx = np.where(G_j < -1e-9)[0]
        inv_G_n = (-1.0 / G_j[n_idx])[:, None]
        t_n = np.min(H_global[n_idx] * inv_G_n, axis=0)
        spans[:, j] = t_p + t_n
    spreads = np.sqrt(np.sum(spans ** 2, axis=1) / 12.0)
    return float(np.mean(spreads))

if __name__ == "__main__":
    pca_basis = np.random.randn(31, 8)
    tasks = [(np.random.randn(3, 31), pca_basis) for _ in range(200)]
    
    t0 = time.time()
    with Pool(processes=12) as pool:
        res = pool.map(eval_single, tasks)
    t1 = time.time()
    print(f"200 evaluations with 12 workers took {t1 - t0:.2f} s ({(t1 - t0)/200*1000:.2f} ms per eval)")
    print(f"Projected time for 7,878 evaluations: {((t1 - t0)/200) * 7878:.1f} seconds ({((t1 - t0)/200) * 7878 / 60:.2f} minutes)!")
