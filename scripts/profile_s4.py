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

    engine = PolytopeGeometryEngine()
    G_cond, h_cond = engine.condition_constraints(G, h)
    box_res = engine.compute_bounding_box(G_cond, h_cond)
    c_cheb, r_in = engine.compute_chebyshev_center(G_cond, h_cond)
    c_box = box_res["c_box"]
    s_box = box_res["s_box"]
    max_box = G_cond @ c_box + np.abs(G_cond) @ s_box
    intersect_box = max_box >= (h_cond - 1e-7)
    n_intersect = np.sum(intersect_box)
    print(f"Total constraints: {len(G_cond)}, Intersect bounding box: {n_intersect} ({n_intersect / len(G_cond) * 100:.2f}%)")

    slacks_cheb = h_cond - G_cond @ c_cheb
    sorted_indices = np.argsort(slacks_cheb)
    
    print(f"Top 10 slacks: {slacks_cheb[sorted_indices[:10]]}")
    
    # Try different initial_k
    for k in [100, 200, 300]:
        W = sorted_indices[:k]
        G_W = G_cond[W]
        h_W = h_cond[W]
        G_scaled = G_W * s_box
        h_scaled = h_W - G_W @ c_cheb
        halfspaces = np.hstack([G_scaled, -h_scaled[:, None]])
        interior_pt = np.zeros(5)

        t0 = time.perf_counter()
        try:
            hs = HalfspaceIntersection(halfspaces, interior_pt)
            t1 = time.perf_counter()
            V = hs.intersections * s_box + c_cheb
            print(f"k={k}: Qhull succeeded in {t1 - t0:.3f} s, vertices={len(V)}, dual facets={len(hs.dual_facets) if hasattr(hs, 'dual_facets') else None}")
            
            # Check violation time using candidate filter
            t2 = time.perf_counter()
            v_min = np.min(V, axis=0)
            v_max = np.max(V, axis=0)
            c_V = (v_min + v_max) / 2.0
            s_V = (v_max - v_min) / 2.0
            
            # Fast filter in 5 ms:
            upper_bound = G_cond @ c_V + np.abs(G_cond) @ s_V
            candidates = np.where(upper_bound > h_cond + 1e-7)[0]
            t_filter = time.perf_counter()
            print(f"  Fast filter took {t_filter - t2:.4f} s: {len(candidates)} candidate constraints out of {len(G_cond)}")
            
            if len(candidates) > 0:
                G_cand = G_cond[candidates]
                h_cand = h_cond[candidates]
                # Check all vertices against only candidate constraints!
                # Even with 3000 vertices: (len(candidates) x 3000)
                # Chunk candidate checks:
                chunk_size = 1000
                max_viol = -np.inf
                bad_cand_indices = set()
                for v_start in range(0, len(V), chunk_size):
                    v_ch = V[v_start : v_start + chunk_size]
                    viol_ch = G_cand @ v_ch.T - h_cand[:, None]
                    ch_max = np.max(viol_ch)
                    if ch_max > max_viol:
                        max_viol = ch_max
                    bad_mask = np.any(viol_ch > 1e-7, axis=1)
                    bad_cand_indices.update(np.where(bad_mask)[0])
                t_exact = time.perf_counter()
                print(f"  Exact check on candidates took {t_exact - t_filter:.4f} s: {len(bad_cand_indices)} violated constraints, max viol={max_viol:.6e}")
                print(f"  TOTAL CHECK TIME: {t_exact - t2:.4f} s!")
        except QhullError as e:
            print(f"k={k}: Qhull failed: {e}")

if __name__ == "__main__":
    main()
