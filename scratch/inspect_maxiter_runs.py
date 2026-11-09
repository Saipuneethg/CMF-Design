import os
import json
import numpy as np
import pandas as pd

base_dir = r"c:\Users\saipu\OneDrive\Desktop\fiber polytopes-2"
exp_dir = os.path.join(base_dir, "camera comparisions", "reports", "Experiment_4_Initialization_Independence_20261031_221323")
data_dir = os.path.join(exp_dir, "Data")
final_curves_dir = os.path.join(exp_dir, "Final_Optimized_Curves")

df_results = pd.read_csv(os.path.join(data_dir, "final_ambiguity_results.csv"))
maxiter_runs = df_results[df_results["Status"] == "MAXITER"]

print("Identified MAXITER runs:")
for idx, r in maxiter_runs.iterrows():
    print(f"  - {r['Initialization']}: Iters={r['Iterations']}, FunEvals={r['Function_Evaluations']}, Final Amb={r['Final_Test_Ambiguity_Mean']:.4f}, DeltaE00={r['Final_Test_DeltaE00_Mean']:.2f}")
