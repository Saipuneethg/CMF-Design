import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.metamer_solver import MetamerSolver

def main():
    artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
    solver = MetamerSolver(artifacts_dir=artifacts_dir)
    G = solver.G
    print(f"G shape: {G.shape}")
    
    # Check norms
    norms = np.linalg.norm(G, axis=1)
    zero_rows = np.sum(norms < 1e-12)
    print(f"Zero norm rows: {zero_rows}")
    
    # Normalized rows
    non_zero = norms > 1e-12
    G_norm = G[non_zero] / norms[non_zero, None]
    
    # Unique rows with rounding 1e-6
    rounded = np.round(G_norm, decimals=5)
    unique_rows, counts = np.unique(rounded, axis=0, return_counts=True)
    print(f"Unique directions at 5 decimals: {len(unique_rows)} out of {len(G)}")
    
    rounded6 = np.round(G_norm, decimals=6)
    unique_rows6 = np.unique(rounded6, axis=0)
    print(f"Unique directions at 6 decimals: {len(unique_rows6)} out of {len(G)}")

if __name__ == "__main__":
    main()
