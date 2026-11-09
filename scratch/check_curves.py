import os
import numpy as np

curves_dir = "camera comparisions/reports/Experiment_4_Initialization_Independence_20261031_221323/Final_Optimized_Curves"
for f in sorted(os.listdir(curves_dir)):
    if f.endswith(".npy"):
        M = np.load(os.path.join(curves_dir, f))
        print(f"{f:50s} shape: {M.shape}, row sums: {np.sum(M, axis=1)}")
