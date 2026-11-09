import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metamer_solver import MetamerSolver
from src.spectral_bounds import SpectralBoundsCalculator

def main():
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    solver = MetamerSolver(artifacts_dir=artifacts_dir)
    target_rgb = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    fiber = solver.construct_fiber(target_rgb)
    z0 = fiber["z0"]
    h = fiber["h"]
    G = fiber["G"]
    N = fiber["N"]

    print(f"Testing 31-band LP bounds on full G: {G.shape}")
    bounds_calc = SpectralBoundsCalculator(solver.pca_basis, solver.pca_mean, N, G)
    
    # Test 1 band first
    t0 = time.perf_counter()
    res1 = bounds_calc.compute_wavelength_bound(15, z0, h)
    t1 = time.perf_counter()
    print(f"Band 15 (550nm): min={res1['exact_min']:.4f}, max={res1['exact_max']:.4f}, span={res1['exact_range']:.4f} in {t1 - t0:.3f} s")

    # Let's test filtering G for LP bounds:
    # Notice G is 714,340 x 5.
    # How many constraints are near-active or within bounding box?
    c_cheb = np.zeros(5) # or chebyshev center
    # If we condition G by dropping redundant or bounding box filter:
    
if __name__ == "__main__":
    main()
