import os
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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

    bounds_calc = SpectralBoundsCalculator(solver.pca_basis, solver.pca_mean, N, G)
    key_bands = [0, 5, 10, 15, 20, 25, 30] # 400, 450, 500, 550, 600, 650, 700 nm

    print(f"Testing 7 key bands in ThreadPoolExecutor(max_workers=4)...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(bounds_calc.compute_wavelength_bound, w, z0, h) for w in key_bands]
        results = [f.result() for f in futures]
    t1 = time.perf_counter()
    print(f"7 key bands completed in {t1 - t0:.2f} s ({len(results)} bounds)")
    for r in results:
        print(f"  {r['wavelength_nm']} nm: [{r['exact_min']:.4f}, {r['exact_max']:.4f}], span={r['exact_range']:.4f}")

if __name__ == "__main__":
    main()
