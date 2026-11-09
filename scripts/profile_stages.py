import os
import sys
import time
import numpy as np

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

    print(f"Raw G: {G.shape}")
    t0 = time.perf_counter()
    engine = PolytopeGeometryEngine(time_budget=600.0)
    G_cond, h_cond = engine.condition_constraints(G, h)
    t1 = time.perf_counter()
    print(f"S1 Conditioning: {G_cond.shape} in {t1 - t0:.3f} s")

    t0 = time.perf_counter()
    box_res = engine.compute_bounding_box(G_cond, h_cond)
    t1 = time.perf_counter()
    print(f"S2 Bounding Box: widths = {box_res['widths']} in {t1 - t0:.3f} s")

    t0 = time.perf_counter()
    c_cheb, r_in = engine.compute_chebyshev_center(G_cond, h_cond)
    t1 = time.perf_counter()
    print(f"S3 Chebyshev: r_in = {r_in:.6e}, c_cheb = {c_cheb} in {t1 - t0:.3f} s")

    # Inspect slacks at Chebyshev center
    slacks = h_cond - G_cond @ c_cheb
    print(f"Min slack: {np.min(slacks):.6e}, Max slack: {np.max(slacks):.6e}")
    # How many constraints have slack < 1e-4, < 1e-3, < 1e-2, < 0.1?
    print(f"Slack < 1e-4: {np.sum(slacks < 1e-4)}")
    print(f"Slack < 1e-3: {np.sum(slacks < 1e-3)}")
    print(f"Slack < 1e-2: {np.sum(slacks < 1e-2)}")
    print(f"Slack < 0.05: {np.sum(slacks < 0.05)}")
    print(f"Slack < 0.1:  {np.sum(slacks < 0.1)}")

if __name__ == "__main__":
    main()
