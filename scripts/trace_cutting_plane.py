import os
import sys
import time
import numpy as np
from scipy.spatial import HalfspaceIntersection, QhullError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metamer_solver import MetamerSolver
from src.fiber_geometry import PolytopeGeometryEngine

def main():
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    solver = MetamerSolver(artifacts_dir=artifacts_dir)
    target_rgb = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    fiber = solver.construct_fiber(target_rgb)
    G = fiber["G"]
    h = fiber["h"]

    engine = PolytopeGeometryEngine(time_budget=600.0)
    G_cond, h_cond = engine.condition_constraints(G, h)
    
    # S3 Chebyshev center directly (faster than bounding box first)
    t0 = time.perf_counter()
    c_cheb, r_in = engine.compute_chebyshev_center(G_cond, h_cond)
    print(f"Chebyshev center in {time.perf_counter() - t0:.2f} s: r_in={r_in:.6e}")
    
    # Bounding box
    t0 = time.perf_counter()
    box_res = engine.compute_bounding_box(G_cond, h_cond)
    print(f"Bounding box in {time.perf_counter() - t0:.2f} s: widths={box_res['widths']}")
    s_box = box_res["s_box"]

    slacks_cheb = h_cond - G_cond @ c_cheb
    sorted_indices = np.argsort(slacks_cheb)

    # Let's see cutting plane progress
    initial_k = 500
    W = list(sorted_indices[:initial_k])
    
    for it in range(15):
        t_it = time.perf_counter()
        G_W = G_cond[W]
        h_W = h_cond[W]
        G_scaled = G_W * s_box
        h_scaled = h_W - G_W @ c_cheb
        halfspaces = np.hstack([G_scaled, -h_scaled[:, None]])
        interior_pt = np.zeros(5)

        try:
            hs = HalfspaceIntersection(halfspaces, interior_pt, qhull_options="Q12")
        except QhullError as e:
            try:
                # retry with QJ (joggle) or Q12 Qx
                hs = HalfspaceIntersection(halfspaces, interior_pt, qhull_options="QJ")
            except QhullError as e2:
                print(f"Iter {it}: Qhull error: {e2}")
                break

        V_scaled = hs.intersections
        V = V_scaled * s_box + c_cheb
        nV = len(V)
        
        # Check violations with small memory footprint
        t_chk = time.perf_counter()
        max_viol = -np.inf
        violated_rows = set()
        chunk_size = 100
        
        for v_start in range(0, nV, chunk_size):
            v_chunk = V[v_start : v_start + chunk_size]
            viol_chunk = G_cond @ v_chunk.T
            viol_chunk -= h_cond[:, None]
            c_max = np.max(viol_chunk)
            if c_max > max_viol:
                max_viol = c_max
            bad_mask = np.any(viol_chunk > 1e-6, axis=1)
            bad_indices = np.where(bad_mask)[0]
            if len(bad_indices) > 0:
                max_per_row = np.max(viol_chunk[bad_indices], axis=1)
                top_bad = bad_indices[np.argsort(max_per_row)[-50:]]
                violated_rows.update(top_bad)

        t_end = time.perf_counter()
        print(f"Iter {it+1}: |W|={len(W)}, nV={nV}, max_viol={max_viol:.6e}, bad_rows={len(violated_rows)}, time={t_end - t_it:.2f}s (chk {t_end - t_chk:.2f}s)")
        
        if max_viol <= 1e-6 or len(violated_rows) == 0:
            print("CONVERGED!")
            break
            
        W = list(set(W).union(violated_rows))

if __name__ == "__main__":
    main()
