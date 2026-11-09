import os
import sys
import time
import numpy as np

# Ensure root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metamer_solver import MetamerSolver
from src.fiber_geometry import PolytopeGeometryEngine

def main():
    print("=== Testing Real Fiber Geometry for RGB [0.5, 0.5, 0.5] ===")
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    solver = MetamerSolver(artifacts_dir=artifacts_dir)
    
    target_rgb = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    fiber = solver.construct_fiber(target_rgb)
    if not fiber["feasible"]:
        print("ERROR: RGB [0.5, 0.5, 0.5] reported infeasible!")
        return

    G = fiber["G"]
    h = fiber["h"]
    N = fiber["N"]
    z0 = fiber["z0"]
    W_mat = solver.pca_basis @ N

    print(f"Original G shape: {G.shape}, h shape: {h.shape}")
    print(f"Fiber dimension: {fiber['fiber_dimension']}, rank: {fiber['constraint_rank']}")

    engine = PolytopeGeometryEngine(time_budget=300.0)
    t0 = time.perf_counter()
    res = engine.execute_pipeline(
        G, h, z0=z0, B=solver.pca_basis, mu=solver.pca_mean, N=N
    )
    total_time = time.perf_counter() - t0

    print(f"\n--- SUCCESS in {total_time:.3f} s ---")
    print(f"Conditioning: original={res['conditioning']['original_rows']}, conditioned={res['conditioning']['conditioned_rows']}, active={res['conditioning']['active_rows']}")
    print(f"n_vertices: {res['geometry']['n_vertices']}")
    print(f"n_facets: {res['geometry']['n_facets']}")
    print(f"volume_5d: {res['geometry']['volume_5d']:.6e}")
    print(f"surface_area: {res['geometry']['surface_area_4d']:.6e}")
    print(f"inradius: {res['chebyshev']['inradius']:.6e}")
    print(f"diameter: {res['extents']['diameter']:.6e}")

    print("\n--- INTEGRITY CHECKS ---")
    all_pass = True
    for check_id, chk in res["integrity_checks"].items():
        status = chk["status"]
        print(f"  {check_id}: {status} - {chk['name']} (measured: {chk.get('measured')}, threshold: {chk.get('threshold')})")
        if status == "FAIL":
            all_pass = False

    if all_pass:
        print("\nALL INTEGRITY CHECKS PASSED (or WARN, 0 FAIL)!")
    else:
        print("\nWARNING: Some checks FAILED!")

if __name__ == "__main__":
    main()
