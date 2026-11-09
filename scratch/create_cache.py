import os
import numpy as np

artifacts_dir = "artifacts"
A_red = np.load(os.path.join(artifacts_dir, "A_red.npy"))
b_red = np.load(os.path.join(artifacts_dir, "b_red.npy"))
sub_idx = np.linspace(0, len(A_red) - 1, 40000, dtype=int)
A_sub = A_red[sub_idx].astype(np.float32)
b_sub = b_red[sub_idx].astype(np.float32)

cache_dir = "scratch/precomputed_cache"
os.makedirs(cache_dir, exist_ok=True)
np.save(os.path.join(cache_dir, "A_sub.npy"), A_sub)
np.save(os.path.join(cache_dir, "b_sub.npy"), b_sub)

# Precompute H_train
pca_mean = np.load(os.path.join(artifacts_dir, "pca_mean.npy")).astype(np.float64)
pca_basis = np.load(os.path.join(artifacts_dir, "pca_basis.npy")).astype(np.float64)
Z_train = np.load("results/global_cmf_reoptimization_20261028_205403/data/Z_train.npy").astype(np.float64)
H_train = np.maximum(b_sub[:, None] - A_sub @ Z_train.T.astype(np.float32), 1e-12)
np.save(os.path.join(cache_dir, "H_train.npy"), H_train)

print(f"Precomputed cache saved! A_sub: {A_sub.shape}, H_train: {H_train.shape}")
