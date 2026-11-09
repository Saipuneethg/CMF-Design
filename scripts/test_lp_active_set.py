import os
import sys
import time
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metamer_solver import MetamerSolver

def main():
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    solver = MetamerSolver(artifacts_dir=artifacts_dir)
    target_rgb = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    fiber = solver.construct_fiber(target_rgb)
    z0 = fiber["z0"]
    h = fiber["h"]
    G = fiber["G"]
    N = fiber["N"]
    d = G.shape[1]

    # Let's test solving LP with slacks at Chebyshev center
    # Find Chebyshev center
    # LP for Chebyshev:
    norms = np.linalg.norm(G, axis=1)
    # Filter zero norm rows
    valid = norms > 1e-12
    G_v = G[valid]
    h_v = h[valid]
    norms_v = norms[valid]

    t0 = time.perf_counter()
    # Let's see: how fast is Chebyshev LP?
    # Maximize r s.t. G_i alpha + norms_i r <= h_i
    # HiGHS with 714k rows:
    c_cheb_obj = np.zeros(d + 1)
    c_cheb_obj[-1] = -1.0 # maximize r
    A_cheb = np.hstack([G_v, norms_v[:, None]])
    res_cheb = linprog(c_cheb_obj, A_ub=A_cheb, b_ub=h_v, bounds=[(None, None)]*d + [(0, None)], method='highs')
    c_cheb = res_cheb.x[:d]
    r_in = res_cheb.x[-1]
    t1 = time.perf_counter()
    print(f"Chebyshev center in {t1 - t0:.3f} s: r_in={r_in:.6e}")

    # Compute slacks at c_cheb
    slacks = h - G @ c_cheb
    # Sort constraints by slack
    sorted_idx = np.argsort(slacks)
    print(f"Sorted top 1000 slacks range: [{slacks[sorted_idx[0]]:.6e}, {slacks[sorted_idx[1000]]:.6e}]")

    # If we take top K constraints (e.g. 5000 constraints with smallest slacks):
    for K in [1000, 2000, 5000]:
        cand = sorted_idx[:K]
        G_cand = G[cand]
        h_cand = h[cand]

        # Solve band 15 on G_cand
        c_obj = (solver.pca_basis[15] @ N)
        t_lp0 = time.perf_counter()
        res_min = linprog(c_obj, A_ub=G_cand, b_ub=h_cand, bounds=[(None, None)]*d, method='highs')
        t_lp1 = time.perf_counter()

        print(f"K={K}: status={res_min.status}, message={res_min.message}")
        if res_min.success:
            alpha_opt = res_min.x
            viol = np.max(G @ alpha_opt - h)
            print(f"  HiGHS took {t_lp1 - t_lp0:.4f} s, obj={res_min.fun:.6e}, max violation across all 714k: {viol:.6e}")

if __name__ == "__main__":
    main()
